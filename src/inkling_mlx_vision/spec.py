"""Exact stochastic tree speculative sampling (SpecInfer / SpecTr), hybrid N-gram || DSpark trees,
Token Recycling, and DSpark Markov-state carryover for Inkling-2-Mini on Apple Silicon.

Theory (Exact Sequential-Residual Multi-Draft Tree Verification)
----------------------------------------------------------------
At any tree node `u` with children `ch_1, ..., ch_m` where each child `ch_j` was proposed from an
arbitrary distribution `q_j` (either a full/shortlist draft distribution `float32[S]` or a Dirac
delta `1[x == c_j]` for deterministic top-k / N-gram / recycled drafts):

    p <- p_u                                        (exact target distribution at node u)
    for j = 1..m:
        accept ch_j with prob min(1, p(c_j) / q_j(c_j))
        if accepted: move to child ch_j and repeat
        else:        p <- norm(max(0, p - q_j))     (exact residual target distribution)
    if all children rejected: sample bonus token from final residual p

By induction over j and tree depth, for EVERY token `x in V`:
    P(emit x at node u) = p_u(x)
at any `temp > 0, top_p, top_k`, even when `q_2` is conditioned on `c_1` (sampling without
replacement: `q_2(x) = q_1(x) * 1[x != c_1] / (1 - q_1(c_1))`).

Crucially, unlike naive Dirac-only trees (which drop depth-2+ acceptance to `p(top1)` at `temp > 0`),
carrying `q_edges` from DSpark gives Branch A the full `1 - TVD(p, q)` stochastic acceptance at EVERY
depth AND adds a second without-replacement stochastic Branch B at root — strictly dominating both
1D stochastic chains and Dirac-only trees.

GPU efficiency: the entire tree acceptance walk runs on GPU in <= 8 scalar ops so the `[V=200058]`
residual and `mx.random.categorical` execute ONCE for the single terminal node `rej_node` (1 GPU sync).
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import mlx.core as mx
import numpy as np
from runtime_mlx.inkling import generate as _gen_mod
from runtime_mlx.inkling import model as _model_mod
from runtime_mlx.inkling.drafter import TREE_TOPOLOGIES
from runtime_mlx.inkling.fast_kernels import make_tree_spec

# --------------------------------------------------------------------------- Helpers


def _sample_from_probs(probs: mx.array) -> mx.array:
    """Sample integer index along the last axis from a normalized probability distribution."""
    logp = mx.where(probs > 0.0, mx.log(mx.maximum(probs, 1e-38)), -float("inf"))
    return mx.random.categorical(logp, axis=-1).astype(mx.int32)


def _zero_and_norm(probs_1d: mx.array, exclude_idx: mx.array) -> mx.array:
    """Zero out `exclude_idx` (if in `[0, S)`) from `probs_1d` and renormalize (falls back to `probs_1d` if sum==0)."""
    S = int(probs_1d.shape[0])
    rem = mx.where(mx.arange(S, dtype=mx.int32) == exclude_idx, 0.0, probs_1d)
    s = mx.sum(rem)
    return mx.where(s > 0.0, rem / mx.maximum(s, 1e-38), probs_1d)


def _residual_dist(p: mx.array, q: mx.array | None, ord_idx: mx.array) -> mx.array:
    """Exact normalized residual `norm(max(0, p - q))` for shortlist `q[S]` or Dirac `q=None` at `ord_idx`."""
    V = int(p.shape[0])
    if q is None:
        res = mx.where(mx.arange(V, dtype=mx.int32) == ord_idx, 0.0, p)
    else:
        S = int(q.shape[0])
        if S == V:
            res = mx.maximum(p - q, 0.0)
        else:
            res = mx.concatenate([mx.maximum(p[:S] - q, 0.0), p[S:]], axis=0)
    s = mx.sum(res)
    return mx.where(s > 0.0, res / mx.maximum(s, 1e-38), p)


def _edge_accept(p_cur: mx.array, q: mx.array | None, c: mx.array, u_val: mx.array) -> mx.array:
    """Strict-`<` exact acceptance predicate (`p_c == 0` is never accepted)."""
    p_c = p_cur[c]
    if q is None:
        return u_val < p_c
    S = int(q.shape[0])
    in_s = c < S
    q_c = mx.where(in_s, q[mx.minimum(c, S - 1)], 0.0)
    return (q_c > 0.0) & ((u_val * q_c) < p_c)


# --------------------------------------------------------------------------- Exact Stochastic Tree Verification


def verify_tree_specinfer_gpu(
    ordered_logits: mx.array,
    order_mx: mx.array,
    inv_order: mx.array,
    draft_ids: mx.array,
    tree_spec: dict,
    *,
    q_edges: Sequence[mx.array | None] | None = None,
    cap: float | None,
    temp: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> tuple[mx.array, mx.array]:
    """Zero-sync GPU SpecInfer exact stochastic tree verification.

    Supports arbitrary trees (`T <= 8`) with:
      - `q_edges=None`: all edges Dirac (deterministic top-k / N-gram proposals),
      - `q_edges=[q_1, ..., q_{T-1}]`: per-edge `float32[S]` draft distributions OR `None` (mixed hybrid trees!).

    Computes the `[V]` residual and `mx.random.categorical` ONLY ONCE for the single terminal node `rej_node`.
    Returns `(reached_vec: bool[T], bonus_tok: int32 scalar)`.
    """
    if ordered_logits.ndim == 3:
        ordered_logits = ordered_logits[0]
    T, V = int(ordered_logits.shape[0]), int(ordered_logits.shape[1])
    p_ord = _model_mod.logits_to_probs(
        _model_mod.softcap(ordered_logits, cap), temp=temp, top_p=top_p, top_k=top_k
    )  # [T, V]
    ids = draft_ids.reshape(T).astype(mx.int32)
    child_ord = inv_order[ids[1:]].astype(mx.int32)  # [T - 1]
    u_rand = mx.random.uniform(shape=(T - 1,), dtype=mx.float32)
    children_tbl = tree_spec["children"]

    # 1. Per-edge acceptance booleans + precomputed final residuals for multi-child nodes (typically only root 0).
    acc: list[mx.array] = [mx.array(False)] * (T - 1)
    multi_res: dict[int, mx.array] = {}
    for u in range(T):
        kids = children_tbl[u]
        if not kids:
            continue
        if len(kids) == 1:
            ch = kids[0]
            q = None if q_edges is None else q_edges[ch - 1]
            acc[ch - 1] = _edge_accept(p_ord[u], q, child_ord[ch - 1], u_rand[ch - 1])
        else:
            p_cur = p_ord[u]
            for ch in kids:
                c = child_ord[ch - 1]
                q = None if q_edges is None else q_edges[ch - 1]
                acc[ch - 1] = _edge_accept(p_cur, q, c, u_rand[ch - 1])
                p_cur = _residual_dist(p_cur, q, c)
            multi_res[u] = p_cur

    # 2. Topological GPU tree walk (<= 8 scalar ops) -> reached_vec[T] and unique terminal node rej_node.
    reached: list[mx.array] = [mx.array(True)] + [mx.array(False)] * (T - 1)
    is_stop: list[mx.array] = [mx.array(False)] * T
    for u in range(T):
        still_open = reached[u]
        for ch in children_tbl[u]:
            a = acc[ch - 1]
            reached[ch] = still_open & a
            still_open = still_open & (~a)
        is_stop[u] = still_open

    reached_vec = mx.stack(reached, axis=0)  # bool[T]
    rej_node = mx.argmax(mx.stack(is_stop, axis=0).astype(mx.int32)).astype(mx.int32)

    # 3. Build the SINGLE `[V]` bonus distribution at `rej_node`.
    p_row = p_ord[rej_node]
    dist = p_row  # leaf fallback

    # 3a. Single-child Dirac nodes
    dirac_nodes = [
        (u, children_tbl[u][0])
        for u in range(T)
        if len(children_tbl[u]) == 1 and (q_edges is None or q_edges[children_tbl[u][0] - 1] is None)
    ]
    if dirac_nodes:
        is_d_arr = [False] * T
        ch_idx_arr = [0] * T
        for u, ch in dirac_nodes:
            is_d_arr[u] = True
            ch_idx_arr[u] = ch - 1
        is_dirac = mx.array(is_d_arr, dtype=mx.bool_)[rej_node]
        d_ord = child_ord[mx.array(ch_idx_arr, dtype=mx.int32)[rej_node]]
        dist = mx.where(is_dirac, _residual_dist(p_row, None, d_ord), dist)

    # 3b. Single-child stochastic `q_edges` nodes
    q_nodes = [
        (u, children_tbl[u][0], q_edges[children_tbl[u][0] - 1])
        for u in range(T)
        if len(children_tbl[u]) == 1 and q_edges is not None and q_edges[children_tbl[u][0] - 1] is not None
    ]
    if q_nodes:
        S = int(q_nodes[0][2].shape[0])
        is_q_arr = [False] * T
        q_rows = [mx.zeros((S,), dtype=mx.float32)] * T
        ch_idx_arr = [0] * T
        for u, ch, q_u in q_nodes:
            is_q_arr[u] = True
            q_rows[u] = q_u
            ch_idx_arr[u] = ch - 1
        is_q = mx.array(is_q_arr, dtype=mx.bool_)[rej_node]
        q_sel = mx.stack(q_rows, axis=0)[rej_node]
        c_sel = child_ord[mx.array(ch_idx_arr, dtype=mx.int32)[rej_node]]
        dist = mx.where(is_q, _residual_dist(p_row, q_sel, c_sel), dist)

    # 3c. Multi-child nodes (already have exact post-all-children residual in `multi_res`)
    for u, m_res in multi_res.items():
        dist = mx.where(rej_node == u, m_res, dist)

    bonus_ord = _sample_from_probs(dist)
    bonus_tok = order_mx[bonus_ord].astype(mx.int32)
    return reached_vec, bonus_tok


def verify_tree_sample_gpu(
    ordered_logits: mx.array,
    order_mx: mx.array,
    inv_order: mx.array,
    draft_ids: mx.array,
    tree_spec: dict,
    *,
    q_edges: Sequence[mx.array | None] | None = None,
    cap: float | None,
    temp: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> tuple[list[int], int, list[int]]:
    """Synchronous convenience wrapper: 1 GPU sync -> `(accepted_path, bonus_tok, draft_block)`."""
    reached_vec, bonus_tok = verify_tree_specinfer_gpu(
        ordered_logits, order_mx, inv_order, draft_ids, tree_spec,
        q_edges=q_edges, cap=cap, temp=temp, top_p=top_p, top_k=top_k,
    )
    mx.eval(draft_ids, reached_vec, bonus_tok)
    draft_block = np.array(draft_ids, dtype=np.int32).reshape(-1).tolist()
    reached = np.array(reached_vec, dtype=bool).tolist()
    path = [i for i, ok in enumerate(reached) if ok]
    return path, int(bonus_tok), draft_block


# --------------------------------------------------------------------------- Batched Draft Head & Proposers


def make_batched_draft_head_probs(
    lm_head,
    *,
    temp: float,
    top_p: float = 1.0,
    top_k: int = 0,
    max_buckets: int = 5,
    bits: int = 4,
):
    """Return `(probs_fn(h_2d) -> float32[B, S], order_sub: int32[S])` using the cached 32K 4-bit draft head."""
    cap = lm_head.cap
    if not getattr(lm_head, "_fast", False):
        order_mx = lm_head.order_mx

        def _probs_full(h: mx.array) -> mx.array:
            h2 = h.reshape(-1, lm_head.hidden_size)
            raw = lm_head._ordered_logits(h2)
            return _model_mod.logits_to_probs(
                _model_mod.softcap(raw.astype(mx.float32), cap), temp=temp, top_p=top_p, top_k=top_k
            )

        return _probs_full, order_mx

    wq_all, sc_all, bi_all, order_sub, qbits = lm_head._ensure_draft_head(max_buckets, bits)

    def _probs_fast(h: mx.array) -> mx.array:
        raw = mx.quantized_matmul(
            h.reshape(-1, 2304).astype(mx.float16), wq_all, sc_all, bi_all,
            transpose=True, group_size=64, bits=qbits,
        )
        return _model_mod.logits_to_probs(
            _model_mod.softcap(raw.astype(mx.float32), cap), temp=temp, top_p=top_p, top_k=top_k
        )

    return _probs_fast, order_sub


def propose_tree43_sample_gpu(
    drafter,
    anchor_tok: int | mx.array,
    embed_fn,
    head_probs_fn,
    order_sub: mx.array,
    *,
    init_state: mx.array | None = None,
) -> tuple[mx.array, list[mx.array], mx.array]:
    """Zero-sync stochastic `"4-3"` DSpark tree proposal (4 batched head calls, without-replacement root split).

    Returns `(draft_ids[8], q_edges[7], node_states[8, R])`:
      - Branch A (`nodes 1..4`): sampled from `q_a1, q_a2, q_a3, q_a4`.
      - Branch B (`nodes 5..7`): root `b1` sampled from `q_b1 = norm(q_a1 * 1[x != a1])` (without replacement),
        followed by `b2 ~ q_b2, b3 ~ q_b3`.
    """
    anc_mx = (
        anchor_tok.reshape(1).astype(mx.int32)
        if isinstance(anchor_tok, mx.array)
        else mx.array([int(anchor_tok)], dtype=mx.int32)
    )
    anc_emb = embed_fn(anc_mx)[0, 0, :]
    h = drafter._run_3layers(anc_emb)
    z0 = (drafter._zero_state if init_state is None else init_state.reshape(-1).astype(mx.float32))[None, :]

    st1, hp1 = drafter._markov_step(z0, anc_mx, h[1])
    q_a1 = head_probs_fn(hp1)[0]  # [S]
    ord_a1 = _sample_from_probs(q_a1)
    q_b1 = _zero_and_norm(q_a1, ord_a1)
    ord_b1 = _sample_from_probs(q_b1)
    t1 = order_sub[mx.stack([ord_a1, ord_b1])].astype(mx.int32)  # [2]

    st2, hp2 = drafter._markov_step(mx.concatenate([st1, st1], axis=0), t1, h[2])
    q2 = head_probs_fn(hp2)  # [2, S]
    t2 = order_sub[_sample_from_probs(q2)].astype(mx.int32)  # [2]

    st3, hp3 = drafter._markov_step(st2, t2, h[3])
    q3 = head_probs_fn(hp3)  # [2, S]
    t3 = order_sub[_sample_from_probs(q3)].astype(mx.int32)  # [2]

    st4, hp4 = drafter._markov_step(st3[:1], t3[:1], h[4])
    q_a4 = head_probs_fn(hp4)[0]  # [S]
    a4 = order_sub[_sample_from_probs(q_a4)].reshape(1).astype(mx.int32)

    draft_ids = mx.concatenate([anc_mx, t1[:1], t2[:1], t3[:1], a4, t1[1:2], t2[1:2], t3[1:2]], axis=0)
    q_edges = [q_a1, q2[0], q3[0], q_a4, q_b1, q2[1], q3[1]]
    node_states = mx.concatenate([z0, st1, st2[:1], st3[:1], st4, st1, st2[1:2], st3[1:2]], axis=0)
    return draft_ids, q_edges, node_states


def propose_hybrid43_sample_gpu(
    drafter,
    branch_a: Sequence[int],
    embed_fn,
    head_probs_fn,
    order_sub: mx.array,
    inv_order: mx.array,
    *,
    init_state: mx.array | None = None,
) -> tuple[mx.array, list[mx.array | None]]:
    """Zero-sync stochastic Hybrid `"4-3"` proposal: Branch A (N-gram/Recycled, Dirac) || Branch B (DSpark, 3 calls).

    Zeros out `branch_a[1]` from `q_b1` so Branch B explores distinct tokens from Branch A at root.
    """
    ng_mx = mx.array(np.array(list(branch_a)[:5], dtype=np.int32))
    anc_mx = ng_mx[:1]
    anc_emb = embed_fn(anc_mx)[0, 0, :]
    h = drafter._run_3layers(anc_emb)
    z0 = (drafter._zero_state if init_state is None else init_state.reshape(-1).astype(mx.float32))[None, :]

    st1, hp1 = drafter._markov_step(z0, anc_mx, h[1])
    q_b1_raw = head_probs_fn(hp1)[0]
    ng1_ord = inv_order[ng_mx[1]].astype(mx.int32)
    q_b1 = _zero_and_norm(q_b1_raw, ng1_ord)
    b1 = order_sub[_sample_from_probs(q_b1)].reshape(1).astype(mx.int32)

    st2, hp2 = drafter._markov_step(st1, b1, h[2])
    q_b2 = head_probs_fn(hp2)[0]
    b2 = order_sub[_sample_from_probs(q_b2)].reshape(1).astype(mx.int32)

    _, hp3 = drafter._markov_step(st2, b2, h[3])
    q_b3 = head_probs_fn(hp3)[0]
    b3 = order_sub[_sample_from_probs(q_b3)].reshape(1).astype(mx.int32)

    draft_ids = mx.concatenate([ng_mx, b1, b2, b3], axis=0)
    q_edges: list[mx.array | None] = [None, None, None, None, q_b1, q_b2, q_b3]
    return draft_ids, q_edges


def propose_tree43_with_states(drafter, anchor_tok, embed_fn, head_topk_fn, init_state: mx.array | None = None):
    """Greedy/top-k `"4-3"` DSpark tree proposal that also returns per-node Markov states `[8, R]`."""
    anc_mx = (
        anchor_tok.reshape(1).astype(mx.int32)
        if isinstance(anchor_tok, mx.array)
        else mx.array([int(anchor_tok)], dtype=mx.int32)
    )
    anc_emb = embed_fn(anc_mx)[0, 0, :]
    h = drafter._run_3layers(anc_emb)
    z0 = (drafter._zero_state if init_state is None else init_state.reshape(-1).astype(mx.float32))[None, :]
    st1, hp1 = drafter._markov_step(z0, anc_mx, h[1])
    top2 = head_topk_fn(hp1, 2)[0]
    st2, hp2 = drafter._markov_step(mx.concatenate([st1, st1], axis=0), top2, h[2])
    t2 = head_topk_fn(hp2, 1)[:, 0]
    st3, hp3 = drafter._markov_step(st2, t2, h[3])
    t3 = head_topk_fn(hp3, 1)[:, 0]
    st4, hp4 = drafter._markov_step(st3[:1], t3[:1], h[4])
    a4 = head_topk_fn(hp4, 1)[0, :1]
    ids = mx.concatenate([anc_mx, top2[:1], t2[:1], t3[:1], a4, top2[1:2], t2[1:2], t3[1:2]], axis=0)
    node_states = mx.concatenate([z0, st1, st2[:1], st3[:1], st4, st1, st2[1:2], st3[1:2]], axis=0)
    return ids, node_states


def fuse_hybrid_43(ds_tree: mx.array, branch_a: Sequence[int]) -> mx.array:
    """Zero-sync greedy `4-3` hybrid: Branch A (nodes 1..4) = N-gram/recycled chain, Branch B (5..7) = DSpark.

    Branch B takes DSpark's primary `[a1, a2, a3]` unless `a1 == ng1`, in which case it takes `[b1, b2, b3]`.
    """
    ng = mx.array(np.array(list(branch_a)[:5], dtype=np.int32))
    same = ds_tree[1] == ng[1]
    b = mx.where(same, ds_tree[5:8], ds_tree[1:4])
    return mx.concatenate([ng, b.astype(mx.int32)], axis=0)


# --------------------------------------------------------------------------- Token Recycling (verified transitions)


class TokenRecycler:
    """Per-session table of verified `(prev, cur) -> next` transitions from committed target tokens.

    A transition is proposed only after it was observed consistently `min_count` times, keeping recycled
    chains high precision while acting as a zero-ms Branch A source when N-gram misses.
    """

    def __init__(self, capacity: int = 16384, min_count: int = 2):
        self.capacity = int(capacity)
        self.min_count = int(min_count)
        self._bi: dict[tuple[int, int], list[int]] = {}
        self._tail: list[int] = []

    def __len__(self) -> int:
        return len(self._bi)

    def observe(self, toks: Sequence[int]) -> None:
        tail = self._tail
        bi = self._bi
        for t in toks:
            t = int(t)
            if len(tail) >= 2:
                key = (tail[-2], tail[-1])
                ent = bi.get(key)
                if ent is None:
                    bi[key] = [t, 1]
                elif ent[0] == t:
                    ent[1] += 1
                else:
                    ent[0], ent[1] = t, 1
            tail.append(t)
            if len(tail) > 2:
                del tail[0]
        if len(bi) > self.capacity:
            keep = list(bi.items())[-(self.capacity // 2) :]
            self._bi = dict(keep)

    def propose(self, n: int = 4) -> list[int] | None:
        if len(self._tail) < 2:
            return None
        a, b = self._tail[-2], self._tail[-1]
        out: list[int] = []
        for _ in range(n):
            ent = self._bi.get((a, b))
            if ent is None or ent[1] < self.min_count:
                return None
            out.append(ent[0])
            a, b = b, ent[0]
        return out


# --------------------------------------------------------------------------- Extended stream_dspark


@dataclass
class _Plan:
    source: str
    draft: mx.array | None = None
    draft_list: list[int] | None = None
    draft_ord: mx.array | None = None
    q_probs: mx.array | None = None
    q_edges: list[mx.array | None] | None = None
    block_size: int = 1
    node_states: mx.array | None = None
    hybrid_src: str | None = None


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


_UNSET: Any = object()


def stream_dspark_ext(
    model,
    drafter,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 256,
    eos_ids: set[int] | Sequence[int] = (200006,),
    adaptive: bool = True,
    draft_len: int | None = None,
    tree: str | None = None,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    max_logit_gap: float = 0.0,
    min_prob_ratio: float | None = None,
    ngram: bool = False,
    ngram_max_draft: int = 8,
    ngram_min_match: int = 5,
    stats_out: dict[str, Any] | None = None,
    prefill_chunk: Any = _UNSET,
    prompt_cache=None,
    hybrid: bool | None = None,
    recycle: bool | None = None,
    markov_carry: bool | None = None,
    adaptive_tree: bool = False,
) -> Iterator[list[int]]:
    """Superset of stock `stream_dspark` supporting exact SpecInfer stochastic tree verification (`temp > 0`),
    hybrid N-gram || DSpark `4-3` trees, Token Recycling, and DSpark Markov-state carryover."""
    g = _gen_mod
    if prefill_chunk is _UNSET:
        prefill_chunk = g.PREFILL_CHUNK
    if temp < 0.0:
        raise ValueError(f"temp must be >= 0, got {temp}")
    eff_margin = g._compile_margin(max_logit_gap, min_prob_ratio)
    if tree is not None and tree not in TREE_TOPOLOGIES:
        raise ValueError(f"Unknown tree topology {tree!r}; expected one of {tuple(TREE_TOPOLOGIES)}")
    stochastic = temp > 0.0
    if stochastic and eff_margin > 0.0:
        raise ValueError("temp > 0 cannot be combined with max_logit_gap/min_prob_ratio")

    m = g._unwrap_model(model)
    eos = set(int(x) for x in eos_ids)
    tap_ids = drafter.target_layer_ids
    fixed_block = None if draft_len is None else max(2, min(int(draft_len), drafter.block_size))
    cur_block = fixed_block if fixed_block is not None else min(6, drafter.block_size)
    ngram_cap = max(2, min(int(ngram_max_draft), 8))
    tree_spec = make_tree_spec(TREE_TOPOLOGIES[tree]) if tree is not None else None
    tree_parents_mx = mx.array(np.array(TREE_TOPOLOGIES[tree][1:], dtype=np.int32)) if tree is not None else None
    is43 = tree == "4-3"
    use_hybrid = bool(_env_flag("LAMQ_HYBRID_TREE", "1") if hybrid is None else hybrid) and is43 and ngram
    use_recycle = bool(_env_flag("LAMQ_TOKEN_RECYCLE", "1") if recycle is None else recycle) and is43
    use_carry = bool(_env_flag("LAMQ_MARKOV_CARRY", "0") if markov_carry is None else markov_carry) and is43
    carry_decay = float(os.environ.get("LAMQ_MARKOV_CARRY_DECAY", "0.5"))

    draft_head = draft_head_topk = draft_head_sampler = None
    draft_head_probs = order_sub = None
    if tree_spec is not None and stochastic and is43:
        draft_head_probs, order_sub = make_batched_draft_head_probs(m.lm_head, temp=temp, top_p=top_p, top_k=top_k)
        if adaptive_tree:
            draft_head_sampler = m.lm_head.get_draft_head_sampler(temp=temp, top_p=top_p, top_k=top_k)
    elif tree_spec is not None:
        draft_head_topk = m.lm_head.get_draft_head_topk()
    elif stochastic:
        draft_head_sampler = m.lm_head.get_draft_head_sampler(temp=temp, top_p=top_p, top_k=top_k)
    else:
        draft_head = m.lm_head.get_draft_head_argmax()

    old_wire = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    t0 = t1 = t2 = None
    emitted_total = 0
    accepted_per_round: list[int] = []
    round_ms: list[float] = []
    serial_steps = branch_hits = ngram_rounds = ngram_accepted = ngram_first_rejects = 0
    hybrid_rounds = hybrid_a_hits = recycle_rounds = 0
    reused = 0
    try:
        t0 = time.perf_counter()
        if stochastic:
            first_ord_logits, cache, taps, taps_start, reused = g._prefill_prompt(
                m, prompt_ids, ordered=True, prefill_chunk=prefill_chunk, prompt_cache=prompt_cache,
                tap_visits=tap_ids, drafter=drafter,
            )
            first_mx = g._sample_ordered_gpu(
                first_ord_logits[0, -1], m.lm_head.order_mx, cap=m.lm_head.cap, temp=temp, top_p=top_p, top_k=top_k
            )
        else:
            first_mx, cache, taps, taps_start, reused = g._prefill_prompt(
                m, prompt_ids, ordered=False, prefill_chunk=prefill_chunk, prompt_cache=prompt_cache,
                tap_visits=tap_ids, drafter=drafter,
            )
            first_mx = first_mx[0, -1]
        mx.eval(first_mx, taps)
        first = int(first_mx)
        t1 = time.perf_counter()

        drafter.push_context(taps, start_pos=taps_start)
        ngram_idx = g.NgramDraftIndex(prompt_ids) if ngram else None
        if ngram_idx is not None:
            ngram_idx.extend([first])
            ngram_idx._sync_index(max(2, ngram_min_match))
        recycler = TokenRecycler() if use_recycle else None
        if recycler is not None:
            recycler.observe(list(prompt_ids[-4096:]) + [first])
        mx.eval(drafter.f_rnn)
        t2 = time.perf_counter()

        emitted_total = 1
        anchor = first
        yield [first]

        dspark_rounds = 0
        ewma_accept = 3.0
        serial_cooldown = 0
        serial_burst_len = 8
        carry_state: mx.array | None = None
        pending: _Plan | None = None

        def _plan_next_draft() -> _Plan:
            in_cooldown = adaptive and serial_cooldown > 0
            if ngram_idx is not None:
                mn = max(3, min(5, int(ngram_min_match)))
                ng_list = ngram_idx.propose(anchor, max_draft=ngram_cap, min_cont=mn, eos=eos, min_n=mn)
                if ng_list is not None:
                    ng_mx = mx.array(ng_list, dtype=mx.int32)
                    ng_ord = m.lm_head.inv_order[ng_mx[1:]] if stochastic else None
                    return _Plan("ngram", ng_mx, ng_list, ng_ord, None, None, len(ng_list))
            if in_cooldown:
                return _Plan("serial")
            if tree_spec is not None:
                branch_a = None
                hsrc = None
                if use_hybrid and ngram_idx is not None:
                    hb = ngram_idx.propose(anchor, max_draft=5, min_cont=4, eos=eos, min_n=3)
                    if hb is not None and len(hb) == 5:
                        branch_a, hsrc = hb, "ngram"
                if branch_a is None and recycler is not None:
                    rc = recycler.propose(4)
                    if rc is not None and not any(t in eos for t in rc):
                        branch_a, hsrc = [anchor, *rc], "recycle"

                if stochastic and is43:
                    assert draft_head_probs is not None and order_sub is not None
                    if branch_a is not None:
                        d_ids, q_e = propose_hybrid43_sample_gpu(
                            drafter, branch_a, m.embed, draft_head_probs, order_sub, m.lm_head.inv_order,
                            init_state=carry_state if use_carry else None,
                        )
                        return _Plan("hybrid", d_ids, None, None, None, q_e, tree_spec["T"], None, hsrc)
                    if adaptive_tree and draft_head_sampler is not None and ewma_accept >= 3.6:
                        d_ids, d_ord, q_p = drafter.propose_sample_gpu(
                            anchor, m.embed, draft_head_sampler, draft_len=drafter.block_size
                        )
                        return _Plan("dspark_chain", d_ids, None, d_ord, q_p, None, drafter.block_size)
                    d_ids, q_e, node_states = propose_tree43_sample_gpu(
                        drafter, anchor, m.embed, draft_head_probs, order_sub,
                        init_state=carry_state if use_carry else None,
                    )
                    return _Plan("dspark", d_ids, None, None, None, q_e, tree_spec["T"], node_states)

                node_states = None
                if is43 and use_carry and branch_a is None:
                    d_ids, node_states = propose_tree43_with_states(
                        drafter, anchor, m.embed, draft_head_topk, init_state=carry_state
                    )
                else:
                    d_ids = drafter.propose_tree8_gpu(anchor, m.embed, draft_head_topk, topology=tree)
                if branch_a is not None:
                    return _Plan("hybrid", fuse_hybrid_43(d_ids, branch_a), None, None, None, None, tree_spec["T"], None, hsrc)
                return _Plan("dspark", d_ids, None, None, None, None, tree_spec["T"], node_states)

            if stochastic:
                d_ids, d_ord, q_p = drafter.propose_sample_gpu(anchor, m.embed, draft_head_sampler, draft_len=cur_block)
                return _Plan("dspark", d_ids, None, d_ord, q_p, None, cur_block)
            d_ids = drafter.propose_gpu(anchor, m.embed, draft_head, draft_len=cur_block)
            return _Plan("dspark", d_ids, None, None, None, None, cur_block)

        def _async_eval_drafter_and_pending(p: _Plan | None) -> None:
            eval_list: list[mx.array] = [drafter.f_rnn]
            eval_list.extend(x for x in drafter.ctx_k if x is not None)
            eval_list.extend(x for x in drafter.ctx_v if x is not None)
            if p is not None:
                if p.source in ("dspark", "hybrid", "dspark_chain"):
                    eval_list.extend(x for x in (p.draft, p.draft_ord, p.q_probs, p.node_states) if x is not None)
                    if p.q_edges:
                        eval_list.extend(q for q in p.q_edges if q is not None)
                elif p.source == "ngram" and p.draft_ord is not None:
                    eval_list.append(p.draft_ord)
            mx.async_eval(*eval_list)

        while emitted_total < max_tokens and anchor not in eos:
            tr0 = time.perf_counter()
            if pending is None:
                pending = _plan_next_draft()
                if pending.source in ("dspark", "hybrid", "dspark_chain"):
                    init_eval = [x for x in (pending.draft, pending.draft_ord, pending.q_probs) if x is not None]
                    if pending.q_edges:
                        init_eval.extend(q for q in pending.q_edges if q is not None)
                    if init_eval:
                        mx.async_eval(*init_eval)
            plan = pending
            pending = None
            round_source = plan.source
            draft_gpu = plan.draft
            draft_list = plan.draft_list
            block_size = plan.block_size

            if round_source == "serial":
                if stochastic:
                    nxt_ord_logits, cache, taps = m.forward_ordered_logits([anchor], cache=cache, tap_visits=tap_ids)
                    nxt_mx = g._sample_ordered_gpu(
                        nxt_ord_logits[0, -1], m.lm_head.order_mx, cap=m.lm_head.cap, temp=temp, top_p=top_p, top_k=top_k
                    )
                else:
                    nxt_mx, cache, taps = m.forward_argmax([anchor], cache=cache, tap_visits=tap_ids)
                    nxt_mx = nxt_mx[0, -1]
                mx.eval(nxt_mx, taps)
                nxt = int(nxt_mx)
                emitted_total += 1
                anchor = nxt
                serial_steps += 1
                serial_cooldown -= 1
                carry_state = None
                if ngram_idx is not None:
                    ngram_idx.observe([nxt], 0, 1, False)
                if recycler is not None:
                    recycler.observe([nxt])
                drafter.push_context(taps, start_pos=cache.seq_len - 1)
                if emitted_total < max_tokens and anchor not in eos:
                    pending = _plan_next_draft()
                _async_eval_drafter_and_pending(pending)
                yield [nxt]
                continue

            base_pos = cache.seq_len
            cache.begin_spec_verify()
            accepted_path: list[int] | None = None

            if tree_spec is not None and round_source in ("dspark", "hybrid"):
                children_tbl = tree_spec["children"]
                if stochastic:
                    ord_logits, cache, ver_taps = m.forward_all_ordered_logits(
                        draft_gpu, cache=cache, tap_visits=tap_ids, tree_spec=tree_spec
                    )
                    reached_mx, bonus_mx = verify_tree_specinfer_gpu(
                        ord_logits[0], m.lm_head.order_mx, m.lm_head.inv_order, draft_gpu, tree_spec,
                        q_edges=plan.q_edges, cap=m.lm_head.cap, temp=temp, top_p=top_p, top_k=top_k,
                    )
                    mx.eval(draft_gpu, reached_mx, bonus_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                    accepted_path = [i for i, ok in enumerate(np.array(reached_mx, dtype=bool).tolist()) if ok]
                    bonus_tok = int(bonus_mx)
                else:
                    if eff_margin > 0.0:
                        ord_logits, cache, ver_taps = m.forward_all_ordered_logits(
                            draft_gpu, cache=cache, tap_visits=tap_ids, tree_spec=tree_spec
                        )
                        ol = ord_logits[0]
                        best_ord = mx.argmax(ol, axis=-1).astype(mx.int32)
                        ver_preds_mx = m.lm_head.order_mx[best_ord]
                        rows_t = mx.arange(block_size, dtype=mx.int32)
                        max_sc_mx = _model_mod.softcap(ol[rows_t, best_ord].astype(mx.float32), m.lm_head.cap)
                        child_ord = m.lm_head.inv_order[draft_gpu[1:]].astype(mx.int32)
                        child_sc_mx = _model_mod.softcap(ol[tree_parents_mx, child_ord].astype(mx.float32), m.lm_head.cap)
                        mx.eval(draft_gpu, ver_preds_mx, max_sc_mx, child_sc_mx, ver_taps)
                        max_sc = np.array(max_sc_mx, dtype=np.float32).tolist()
                        child_sc = np.array(child_sc_mx, dtype=np.float32).tolist()
                    else:
                        ver_preds_mx, cache, ver_taps = m.forward_all_argmax(
                            draft_gpu, cache=cache, tap_visits=tap_ids, tree_spec=tree_spec
                        )
                        ver_preds_mx = ver_preds_mx[0]
                        mx.eval(draft_gpu, ver_preds_mx, ver_taps)
                        max_sc = child_sc = None
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                    preds = np.array(ver_preds_mx, dtype=np.int32).tolist()
                    accepted_path = [0]
                    cur = 0
                    while True:
                        want = preds[cur]
                        nxt_node = -1
                        for ch in children_tbl[cur]:
                            if draft_block[ch] == want:
                                nxt_node = ch
                                break
                        if nxt_node < 0 and eff_margin > 0.0 and max_sc is not None and child_sc is not None:
                            thresh = max_sc[cur] - eff_margin
                            best_ch_sc = -float("inf")
                            for ch in children_tbl[cur]:
                                sc_v = child_sc[ch - 1]
                                if sc_v >= thresh and sc_v > best_ch_sc:
                                    best_ch_sc = sc_v
                                    nxt_node = ch
                        if nxt_node < 0:
                            break
                        accepted_path.append(nxt_node)
                        cur = nxt_node
                    bonus_tok = preds[accepted_path[-1]]
                k_match = len(accepted_path) - 1
                is_contiguous = accepted_path == list(range(len(accepted_path)))
                if not is_contiguous:
                    branch_hits += 1
                taps_commit = (
                    ver_taps[: len(accepted_path)]
                    if is_contiguous
                    else ver_taps[mx.array(np.array(accepted_path, dtype=np.int32))]
                )
                raw = [draft_block[n] for n in accepted_path[1:]] + [bonus_tok]
                commit_fn = lambda ap=accepted_path, bs=block_size: cache.commit_tree_verify(ap, bs)  # noqa: E731
            elif stochastic:
                ord_logits, cache, ver_taps = m.forward_all_ordered_logits(draft_gpu, cache=cache, tap_visits=tap_ids)
                k_match_mx, final_tok_mx = g.verify_spec_sample_gpu(
                    ord_logits[0], m.lm_head.order_mx, plan.draft_ord, plan.q_probs,
                    cap=m.lm_head.cap, temp=temp, top_p=top_p, top_k=top_k,
                )
                if draft_list is not None:
                    mx.eval(k_match_mx, final_tok_mx, ver_taps)
                    draft_block = draft_list
                else:
                    mx.eval(draft_gpu, k_match_mx, final_tok_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                k_match = int(k_match_mx)
                k_commit = k_match + 1
                taps_commit = ver_taps[:k_commit]
                raw = draft_block[1 : k_match + 1] + [int(final_tok_mx)]
                commit_fn = lambda kc=k_commit, bs=block_size: cache.commit_spec_verify(kc, bs)  # noqa: E731
            elif eff_margin > 0.0:
                ord_logits, cache, ver_taps = m.forward_all_ordered_logits(draft_gpu, cache=cache, tap_visits=tap_ids)
                k_match_mx, bonus_tok_mx = g.verify_margin_greedy_gpu(
                    ord_logits[0], m.lm_head.order_mx, m.lm_head.inv_order, draft_gpu, cap=m.lm_head.cap, margin=eff_margin
                )
                if draft_list is not None:
                    mx.eval(k_match_mx, bonus_tok_mx, ver_taps)
                    draft_block = draft_list
                else:
                    mx.eval(draft_gpu, k_match_mx, bonus_tok_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                k_match = int(k_match_mx)
                k_commit = k_match + 1
                taps_commit = ver_taps[:k_commit]
                raw = draft_block[1 : k_match + 1] + [int(bonus_tok_mx)]
                commit_fn = lambda kc=k_commit, bs=block_size: cache.commit_spec_verify(kc, bs)  # noqa: E731
            else:
                ver_preds_mx, cache, ver_taps = m.forward_all_argmax(draft_gpu, cache=cache, tap_visits=tap_ids)
                ver_preds_mx = ver_preds_mx[0]
                if draft_list is not None:
                    mx.eval(ver_preds_mx, ver_taps)
                    draft_block = draft_list
                else:
                    mx.eval(draft_gpu, ver_preds_mx, ver_taps)
                    draft_block = np.array(draft_gpu, dtype=np.int32).tolist()
                preds = np.array(ver_preds_mx, dtype=np.int32).tolist()
                k_match = 0
                for j in range(1, block_size):
                    if preds[j - 1] == draft_block[j]:
                        k_match += 1
                    else:
                        break
                k_commit = k_match + 1
                taps_commit = ver_taps[:k_commit]
                raw = draft_block[1 : k_match + 1] + [preds[k_match]]
                commit_fn = lambda kc=k_commit, bs=block_size: cache.commit_spec_verify(kc, bs)  # noqa: E731

            chunk: list[int] = []
            for tok in raw:
                if emitted_total >= max_tokens:
                    break
                chunk.append(tok)
                emitted_total += 1
                anchor = tok
                if tok in eos:
                    break

            accepted_per_round.append(len(raw))
            if round_source == "ngram":
                ngram_rounds += 1
                ngram_accepted += k_match
                if k_match == 0:
                    ngram_first_rejects += 1
            else:
                dspark_rounds += 1
                ewma_accept = 0.7 * ewma_accept + 0.3 * len(raw)
                if adaptive:
                    if tree_spec is None and fixed_block is None:
                        cur_block = drafter.block_size if ewma_accept >= 3.6 else min(6, drafter.block_size)
                    if dspark_rounds >= 5 and ewma_accept < 1.35:
                        serial_cooldown = serial_burst_len
                        serial_burst_len = min(16, serial_burst_len * 2)
                        ewma_accept = 2.0

            k_a = 0
            if round_source == "hybrid" and accepted_path is not None:
                hybrid_rounds += 1
                if plan.hybrid_src == "recycle":
                    recycle_rounds += 1
                while k_a + 1 < len(accepted_path) and accepted_path[k_a + 1] == k_a + 1 and k_a < 4:
                    k_a += 1
                if k_a > 0:
                    hybrid_a_hits += 1

            if ngram_idx is not None:
                if round_source == "hybrid" and plan.hybrid_src == "ngram":
                    ngram_idx.observe(raw, k_a, 5, True)
                else:
                    ngram_idx.observe(raw, k_match, block_size, round_source == "ngram")
            if recycler is not None:
                recycler.observe(raw)

            if use_carry and round_source == "dspark" and plan.node_states is not None and accepted_path is not None:
                carry_state = carry_decay * plan.node_states[accepted_path[-1]]
            else:
                carry_state = None

            drafter.push_context(taps_commit, start_pos=base_pos)
            if emitted_total < max_tokens and anchor not in eos:
                pending = _plan_next_draft()
            _async_eval_drafter_and_pending(pending)

            commit_fn()
            round_ms.append((time.perf_counter() - tr0) * 1e3)
            if chunk:
                yield chunk
    finally:
        if stats_out is not None and t0 is not None and t1 is not None and t2 is not None:
            t3 = time.perf_counter()
            decode_s = max(t3 - t2, 1e-9)
            decode_toks = max(emitted_total - 1, 0)
            mean_acc = sum(accepted_per_round) / max(len(accepted_per_round), 1)
            mode_name = f"dspark_tree_{tree}" if tree is not None else ("dspark_adaptive" if adaptive else "dspark")
            if stochastic:
                mode_name = f"{mode_name}_sample"
            elif eff_margin > 0.0:
                mode_name = f"{mode_name}_margin"
            stats_out.update(
                {
                    "mode": mode_name,
                    "prompt_tokens": len(prompt_ids),
                    "prompt_cached_tokens": reused,
                    "tokens": decode_toks,
                    "generated_tokens": emitted_total,
                    "prompt_s": round(t1 - t0, 4),
                    "drafter_prompt_s": round(t2 - t1, 4),
                    "rounds_s": round(decode_s, 4),
                    "decode_tok_s": round(decode_toks / decode_s, 2) if decode_toks else 0.0,
                    "rounds": len(accepted_per_round),
                    "tokens_per_check": round(mean_acc, 2),
                    "branch_hits": branch_hits,
                    "ngram_rounds": ngram_rounds,
                    "ngram_accepted": ngram_accepted,
                    "ngram_first_rejects": ngram_first_rejects,
                    "hybrid_rounds": hybrid_rounds,
                    "hybrid_a_hits": hybrid_a_hits,
                    "recycle_rounds": recycle_rounds,
                    "median_round_ms": round(statistics.median(round_ms), 2) if round_ms else 0.0,
                    "serial_steps": serial_steps,
                    "temp": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "eff_margin": round(eff_margin, 6),
                    "active_gb": round(mx.get_active_memory() / 1e9, 3),
                    "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
                }
            )
        mx.set_wired_limit(old_wire)
