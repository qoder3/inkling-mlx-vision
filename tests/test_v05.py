"""Tests for v0.5.0 upgrades: pre-tokenization char gate, Kernel 6 (2-slot SwiGLU+RHT-1152),
exact SpecInfer stochastic tree verification, Hybrid N-gram||DSpark trees, Token Recycling,
Markov-state carryover, and Quest L1-bounding-box KV page selection."""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from inkling_mlx_vision import accel, quest, server, spec
from runtime_mlx.inkling.drafter import TREE_TOPOLOGIES
from runtime_mlx.inkling.fast_kernels import make_tree_spec


# --------------------------------------------------------------------------- 1. Pre-tokenization char gate


def test_utf8_byte_bound_pre_tokenization_gate():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "assistant",
            "content": "Calling tool.",
            "reasoning_content": "Let me think " * 10,
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "write_file", "arguments": {"path": "a.py", "text": "x" * 500}}}
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:..."}}]},
    ]
    cnt = server.message_char_count(msgs)
    assert cnt > 600
    assert not server.exceeds_char_budget(cnt, ctx=131072)

    # 655K+ chars (~330K-400K tokens of code/JSON) must be rejected before BPE at 131K context (ctx * 5 = 655,360)
    giant = [{"role": "user", "content": "a" * (131072 * 5)}]
    assert server.exceeds_char_budget(server.message_char_count(giant), ctx=131072)
    assert not server.exceeds_char_budget(131072 * 4, ctx=131072)

    # Normalized dict tool arguments must also trigger the gate (not len(dict) == 1!)
    giant_tool = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "edit", "arguments": {"payload": "z" * (131072 * 7)}}}],
        }
    ]
    assert server.exceeds_char_budget(server.message_char_count(giant_tool), ctx=131072)


# --------------------------------------------------------------------------- 2. Kernel 6: 2-slot SwiGLU + RHT-1152


@pytest.mark.parametrize("n_slots", [513, 1024])
@pytest.mark.parametrize("out_dtype", [mx.float16, mx.float32])
def test_parallel_swiglu_rht1152_parity(n_slots: int, out_dtype):
    rng = np.random.default_rng(500 + n_slots)
    y_gu = mx.array((rng.standard_normal((n_slots, 2304)) * 1.5).astype(np.float32))
    signs = mx.array(np.where(rng.uniform(size=(1152,)) > 0.5, 1.0, -1.0).astype(np.float32))
    eids = mx.array(np.where(rng.uniform(size=(n_slots,)) > 0.8, 0xFFFFFFFF, 7).astype(np.uint32))
    mx.eval(y_gu, signs, eids)

    for kw in ({}, {"also_raw": True}, {"eids_flat": eids, "also_raw": True}, {"pad_zero_row": True}):
        ref = accel._ORIG_SWIGLU_RHT1152(y_gu, signs, 4.0, 25.0, out_dtype=out_dtype, **kw)
        got = accel.fast_swiglu_rht_1152(y_gu, signs, 4.0, 25.0, out_dtype=out_dtype, **kw)
        refs = ref if isinstance(ref, tuple) else (ref,)
        gots = got if isinstance(got, tuple) else (got,)
        mx.eval(*refs, *gots)
        if "eids_flat" in kw:
            keep = np.asarray(eids != 0xFFFFFFFF)[:, None]
            r0 = np.asarray(refs[0], dtype=np.float32)[:n_slots]
            g0 = np.asarray(gots[0], dtype=np.float32)[:n_slots]
            assert float(np.max(np.where(keep, np.abs(r0 - g0), 0.0))) == 0.0
            r1 = np.asarray(refs[1], dtype=np.float32)
            g1 = np.asarray(gots[1], dtype=np.float32)
            assert float(np.max(np.abs(r1 - g1))) == 0.0
        else:
            for r, g in zip(refs, gots):
                assert float(np.max(np.abs(np.asarray(r, dtype=np.float32) - np.asarray(g, dtype=np.float32)))) == 0.0


# --------------------------------------------------------------------------- 3. Exact SpecInfer Stochastic Tree Verification


def _softmax_np(logits: np.ndarray, temp: float) -> np.ndarray:
    z = (logits / temp) - np.max(logits / temp)
    e = np.exp(z)
    return e / np.sum(e)


@pytest.mark.parametrize("dup_siblings", [False, True])
def test_verify_tree_specinfer_dirac_exact_distribution(dup_siblings: bool):
    """Exactness proof for Dirac (q_edges=None) multi-branch tree verification, including duplicate siblings."""
    mx.random.seed(2026 + int(dup_siblings))
    V = 12
    temp = 0.8
    tree_spec = make_tree_spec(TREE_TOPOLOGIES["4-3"])
    rng = np.random.default_rng(77)
    # Non-trivial permutation order_mx / inv_order to verify bucket-order indexing
    perm = rng.permutation(V).astype(np.int32)
    inv = np.argsort(perm).astype(np.int32)
    order_mx = mx.array(perm)
    inv_order = mx.array(inv)

    logits_np = rng.standard_normal((8, V)).astype(np.float32) * 1.4
    p0_ord = _softmax_np(logits_np[0], temp)
    p0_vocab = np.zeros(V, dtype=np.float64)
    p0_vocab[perm] = p0_ord

    # Pick top-2 tokens in vocab space for root children (nodes 1 and 5)
    top_v = np.argsort(-p0_vocab)
    a1 = int(top_v[0])
    b1 = a1 if dup_siblings else int(top_v[1])
    draft_ids = mx.array([0, a1, 2, 3, 4, b1, 5, 6], dtype=mx.int32)
    ord_logits = mx.array(logits_np)

    n_trials = 3500
    counts = np.zeros(V, dtype=np.int64)
    branch_b_hits = 0
    for _ in range(n_trials):
        reached_mx, bonus_mx = spec.verify_tree_specinfer_gpu(
            ord_logits, order_mx, inv_order, draft_ids, tree_spec, q_edges=None, cap=None, temp=temp
        )
        mx.eval(reached_mx, bonus_mx)
        reached = np.asarray(reached_mx, dtype=bool).tolist()
        path = [i for i, ok in enumerate(reached) if ok]
        first_tok = int(draft_ids[path[1]]) if len(path) > 1 else int(bonus_mx)
        counts[first_tok] += 1
        if len(path) > 1 and path[1] == 5:
            branch_b_hits += 1

    emp = counts / n_trials
    max_err = float(np.max(np.abs(emp - p0_vocab)))
    assert max_err < 0.025, f"Dirac tree bias detected: max_err={max_err:.4f}, emp={emp}, ref={p0_vocab}"
    if dup_siblings:
        assert branch_b_hits == 0
    else:
        assert abs(branch_b_hits / n_trials - p0_vocab[b1]) < 0.025


@pytest.mark.parametrize("hybrid_mode", [False, True])
def test_verify_tree_specinfer_stochastic_q_exact_distribution(hybrid_mode: bool):
    """Exactness + dominance proof for SpecInfer stochastic `q_edges` (with shortlist S < V and without-replacement split)."""
    mx.random.seed(4040 + int(hybrid_mode))
    V, S = 14, 9
    temp = 0.7
    tree_spec = make_tree_spec(TREE_TOPOLOGIES["4-3"])
    rng = np.random.default_rng(123)
    perm = rng.permutation(V).astype(np.int32)
    inv = np.argsort(perm).astype(np.int32)
    order_mx = mx.array(perm)
    inv_order = mx.array(inv)
    order_sub = order_mx[:S]

    # Target p_0 has mass both inside shortlist [0..S-1] and in tail [S..V-1]
    logits_np = rng.standard_normal((8, V)).astype(np.float32) * 1.1
    p0_ord = _softmax_np(logits_np[0], temp)
    p0_vocab = np.zeros(V, dtype=np.float64)
    p0_vocab[perm] = p0_ord

    # Draft q_0 over shortlist S with a deliberate distribution shift vs p_0
    q_logits = logits_np[0, :S] + rng.standard_normal(S).astype(np.float32) * 0.6
    q1_base = mx.array(_softmax_np(q_logits, temp).astype(np.float32))
    q_dummy = mx.full((S,), 1.0 / S, dtype=mx.float32)
    ord_logits = mx.array(logits_np)

    n_trials = 4000
    counts = np.zeros(V, dtype=np.int64)
    root_accepts = 0
    branch_b_accepts = 0
    ng1_vocab = int(perm[2])  # fixed N-gram Branch A candidate for hybrid mode
    ng1_ord = mx.array(2, dtype=mx.int32)

    for _ in range(n_trials):
        if hybrid_mode:
            q_b1 = spec._zero_and_norm(q1_base, ng1_ord)
            b1_ord = spec._sample_from_probs(q_b1)
            b1_tok = order_sub[b1_ord]
            draft_ids = mx.stack([
                mx.array(0, dtype=mx.int32),
                mx.array(ng1_vocab, dtype=mx.int32),
                mx.array(int(perm[0]), dtype=mx.int32),
                mx.array(int(perm[1]), dtype=mx.int32),
                mx.array(int(perm[3]), dtype=mx.int32),
                b1_tok,
                mx.array(int(perm[4]), dtype=mx.int32),
                mx.array(int(perm[5]), dtype=mx.int32),
            ])
            q_edges = [None, None, None, None, q_b1, q_dummy, q_dummy]
        else:
            a1_ord = spec._sample_from_probs(q1_base)
            q_b1 = spec._zero_and_norm(q1_base, a1_ord)
            b1_ord = spec._sample_from_probs(q_b1)
            draft_ids = mx.stack([
                mx.array(0, dtype=mx.int32),
                order_sub[a1_ord],
                mx.array(int(perm[0]), dtype=mx.int32),
                mx.array(int(perm[1]), dtype=mx.int32),
                mx.array(int(perm[3]), dtype=mx.int32),
                order_sub[b1_ord],
                mx.array(int(perm[4]), dtype=mx.int32),
                mx.array(int(perm[5]), dtype=mx.int32),
            ])
            q_edges = [q1_base, q_dummy, q_dummy, q_dummy, q_b1, q_dummy, q_dummy]

        reached_mx, bonus_mx = spec.verify_tree_specinfer_gpu(
            ord_logits, order_mx, inv_order, draft_ids, tree_spec, q_edges=q_edges, cap=None, temp=temp
        )
        mx.eval(draft_ids, reached_mx, bonus_mx)
        reached = np.asarray(reached_mx, dtype=bool).tolist()
        path = [i for i, ok in enumerate(reached) if ok]
        first_tok = int(draft_ids[path[1]]) if len(path) > 1 else int(bonus_mx)
        counts[first_tok] += 1
        if len(path) > 1:
            root_accepts += 1
            if path[1] == 5:
                branch_b_accepts += 1

    emp = counts / n_trials
    max_err = float(np.max(np.abs(emp - p0_vocab)))
    assert max_err < 0.025, f"SpecInfer stochastic tree bias (hybrid={hybrid_mode}): max_err={max_err:.4f}"
    # Tail tokens outside shortlist [S..V-1] must also match exact target mass!
    tail_vocab = perm[S:]
    assert abs(float(emp[tail_vocab].sum() - p0_vocab[tail_vocab].sum())) < 0.025
    # Branch B must actively rescue rejections!
    assert branch_b_accepts > 50
    if not hybrid_mode:
        # 2-branch SpecInfer acceptance must strictly exceed 1D single-sample acceptance sum(min(p, q))!
        single_branch_acc = float(np.sum(np.minimum(p0_ord[:S], np.asarray(q1_base))))
        assert root_accepts / n_trials > single_branch_acc + 0.01


# --------------------------------------------------------------------------- 4. Hybrid tree fusion & TokenRecycler


def test_hybrid_tree_fusion_and_token_recycler():
    ds_tree = mx.array([10, 20, 21, 22, 23, 30, 31, 32], dtype=mx.int32)
    # Different root token -> Branch B keeps DSpark primary [20, 21, 22]
    f1 = spec.fuse_hybrid_43(ds_tree, [10, 99, 98, 97, 96])
    mx.eval(f1)
    assert np.asarray(f1).tolist() == [10, 99, 98, 97, 96, 20, 21, 22]

    # Colliding root token (ng1 == a1 == 20) -> Branch B switches to DSpark secondary [30, 31, 32]
    f2 = spec.fuse_hybrid_43(ds_tree, [10, 20, 98, 97, 96])
    mx.eval(f2)
    assert np.asarray(f2).tolist() == [10, 20, 98, 97, 96, 30, 31, 32]

    rec = spec.TokenRecycler(capacity=32, min_count=2)
    # 1st occurrence -> not yet confident
    rec.observe([1, 2, 3, 4, 5, 6])
    rec.observe([1, 2])
    assert rec.propose(4) is None
    # 2nd consistent occurrence -> confident 4-token chain
    rec.observe([3, 4, 5, 6, 1, 2])
    assert rec.propose(4) == [3, 4, 5, 6]
    # Conflicting transition resets count
    rec.observe([99, 1, 2])
    assert rec.propose(4) is None
    # Capacity eviction
    for i in range(60):
        rec.observe([100 + i, 200 + i, 300 + i])
    assert len(rec) <= 32


# --------------------------------------------------------------------------- 5. End-to-end synthetic stream_dspark_ext


class _MockLMHead:
    def __init__(self, V: int, H: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden_size = H
        self.cap = 30.0
        self._fast = False
        perm = rng.permutation(V).astype(np.int32)
        self.order_mx = mx.array(perm)
        self.inv_order = mx.array(np.argsort(perm).astype(np.int32))
        self.w = mx.array((rng.standard_normal((V, H)) * 0.4).astype(np.float32))

    def _ordered_logits(self, h: mx.array) -> mx.array:
        return h.reshape(-1, self.hidden_size).astype(mx.float32) @ self.w.T

    def get_draft_head_topk(self):
        def _topk(h: mx.array, k: int = 1) -> mx.array:
            logits = self._ordered_logits(h)
            part = mx.argpartition(-logits, k - 1, axis=-1)[:, :k]
            vals = mx.take_along_axis(logits, part, axis=-1)
            ord_k = mx.argsort(-vals, axis=-1)
            top = mx.take_along_axis(part, ord_k, axis=-1)
            return self.order_mx[top].astype(mx.int32)
        return _topk

    def get_draft_head_sampler(self, temp: float, top_p: float = 1.0, top_k: int = 0):
        probs_fn, order_sub = spec.make_batched_draft_head_probs(self, temp=temp, top_p=top_p, top_k=top_k)

        def _sample(h_1d: mx.array):
            q = probs_fn(h_1d.reshape(1, self.hidden_size))[0]
            o = spec._sample_from_probs(q)
            return order_sub[o].astype(mx.int32), o, q
        return _sample

    def get_draft_head_argmax(self):
        def _arg(h_1d: mx.array) -> mx.array:
            return self.order_mx[mx.argmax(self._ordered_logits(h_1d)[0])].astype(mx.int32)
        return _arg


class _MockCache:
    def __init__(self):
        self.seq_len = 0
        self.tree_commits = 0
        self.spec_commits = 0

    def begin_spec_verify(self):
        pass

    def commit_tree_verify(self, accepted_path: list[int], block_len: int):
        self.seq_len += len(accepted_path)
        self.tree_commits += 1

    def commit_spec_verify(self, k_accept: int, block_len: int):
        self.seq_len += k_accept
        self.spec_commits += 1


class _MockModel:
    def __init__(self, V: int = 32, H: int = 16):
        rng = np.random.default_rng(7)
        self.lm_head = _MockLMHead(V, H, seed=8)
        self._emb = mx.array((rng.standard_normal((V, H)) * 0.5).astype(np.float32))
        self.H = H

    def make_cache(self):
        return _MockCache()

    def embed(self, ids: mx.array) -> mx.array:
        return self._emb[ids.reshape(-1)][None, :, :]

    def forward_ordered_logits(self, ids, cache=None, tap_visits=()):
        ids_mx = ids if isinstance(ids, mx.array) else mx.array(ids, dtype=mx.int32)
        T = int(ids_mx.reshape(-1).shape[0])
        h = self._emb[ids_mx.reshape(-1)]
        logits = self.lm_head._ordered_logits(h)[None, :, :]
        taps = mx.zeros((T, 4, self.H), dtype=mx.float32)
        if cache is None:
            cache = self.make_cache()
        cache.seq_len += T
        if tap_visits:
            return logits, cache, taps
        return logits, cache

    def forward_argmax(self, ids, cache=None, tap_visits=()):
        logits, cache, taps = self.forward_ordered_logits(ids, cache=cache, tap_visits=tap_visits or (5,))
        preds = self.lm_head.order_mx[mx.argmax(logits, axis=-1)].astype(mx.int32)
        if tap_visits:
            return preds, cache, taps
        return preds, cache

    def forward_all_ordered_logits(self, ids, cache=None, tap_visits=(), tree_spec=None):
        ids_mx = ids if isinstance(ids, mx.array) else mx.array(ids, dtype=mx.int32)
        T = int(ids_mx.reshape(-1).shape[0])
        h = self._emb[ids_mx.reshape(-1)]
        logits = self.lm_head._ordered_logits(h)[None, :, :]
        taps = mx.zeros((T, 4, self.H), dtype=mx.float32)
        return logits, cache, taps


class _MockDrafter:
    def __init__(self, H: int = 16, R: int = 8):
        self.block_size = 8
        self.target_layer_ids = (5, 23, 41, 48)
        self.H = H
        self.R = R
        self._zero_state = mx.zeros((R,), dtype=mx.float32)
        self.f_rnn = mx.zeros((H,), dtype=mx.float32)
        self.ctx_k = [None]
        self.ctx_v = [None]
        rng = np.random.default_rng(19)
        self._proj = mx.array((rng.standard_normal((R, H)) * 0.2).astype(np.float32))

    def reset(self):
        pass

    def push_context(self, taps: mx.array, start_pos: int):
        self.f_rnn = mx.mean(taps[-1], axis=0)

    def _run_3layers(self, anc_emb: mx.array, n_toks: int | None = None) -> mx.array:
        n = self.block_size if n_toks is None else n_toks
        return mx.broadcast_to(anc_emb[None, :], (n, self.H))

    def _markov_step(self, state: mx.array, prev_tok: mx.array, h_s: mx.array):
        B = int(prev_tok.shape[0])
        new_st = 0.5 * state + 0.1
        hp = mx.broadcast_to(h_s[None, :], (B, self.H)) + new_st @ self._proj
        return new_st, hp

    def propose_sample_gpu(self, anchor_tok, embed_fn, head_sample_fn, draft_len=None):
        n = self.block_size if draft_len is None else draft_len
        anc = mx.array([int(anchor_tok)], dtype=mx.int32)
        h = self._run_3layers(embed_fn(anc)[0, 0, :], n_toks=n)
        toks, ords, qs = [anc], [], []
        st = self._zero_state[None, :]
        cur = anc
        for s in range(1, n):
            st, hp = self._markov_step(st, cur, h[s])
            t, o, q = head_sample_fn(hp[0])
            cur = t.reshape(1)
            toks.append(cur)
            ords.append(o.reshape(1))
            qs.append(q)
        return mx.concatenate(toks, axis=0), mx.concatenate(ords, axis=0), mx.stack(qs, axis=0)


@pytest.mark.parametrize("adaptive_tree", [False, True])
def test_stream_dspark_ext_end_to_end_synthetic(adaptive_tree: bool):
    mx.random.seed(99 + int(adaptive_tree))
    model = _MockModel(V=32, H=16)
    drafter = _MockDrafter(H=16, R=8)
    # Repeating prompt pattern to exercise N-gram + Hybrid + TokenRecycler paths
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]
    stats: dict = {}
    out: list[int] = []
    for chunk in spec.stream_dspark_ext(
        model,
        drafter,
        prompt,
        max_tokens=24,
        eos_ids=(31,),
        tree="4-3",
        temp=0.6,
        top_p=0.95,
        ngram=True,
        hybrid=True,
        recycle=True,
        markov_carry=True,
        adaptive_tree=adaptive_tree,
        stats_out=stats,
    ):
        out.extend(chunk)
    assert 1 <= len(out) <= 24
    assert all(0 <= t < 32 for t in out)
    assert stats["mode"] == "dspark_tree_4-3_sample"
    assert stats["generated_tokens"] == len(out)
    assert stats["rounds"] >= 1


# --------------------------------------------------------------------------- 6. Quest L1-Bounding-Box KV Page Filtering


def test_quest_page_bounding_box_upper_bound():
    rng = np.random.default_rng(2026)
    n_kv, n_q, D, page = 8, 24, 128, 64
    n_pages = 48
    L = n_pages * page + 15  # includes partial tail!
    K_np = (rng.standard_normal((n_kv, L, D)) * 0.5).astype(np.float16)
    V_np = (rng.standard_normal((n_kv, L, D)) * 0.5).astype(np.float16)
    q_np = (rng.standard_normal((n_q, D)) * 0.5).astype(np.float32)

    # Plant a high-alignment needle key in distant page 20 for KV head 3 (stronger than page Gaussian envelope)
    needle_page = 20
    needle_pos = needle_page * page + 17
    q_target = q_np[3 * 3]  # first query head of KV head 3
    K_np[3, needle_pos, :] = (q_target / (np.linalg.norm(q_target) + 1e-6) * 24.0).astype(np.float16)

    K = mx.array(K_np)
    V = mx.array(V_np)
    q = mx.array(q_np)

    # Test incremental append across unaligned chunk sizes
    idx = quest.QuestPageIndex(page=page)
    pos = 0
    for step in (37, 91, 128, 500, L):
        end = min(L, pos + step)
        if end > pos:
            idx.append(K[:, pos:end, :])
            pos = end
    assert idx.n_tokens == L
    assert idx.n_pages == n_pages

    head_sc, kv_sc = idx.scores(q)
    mx.eval(head_sc, kv_sc)
    h_np = np.asarray(head_sc, dtype=np.float32)
    kv_np = np.asarray(kv_sc, dtype=np.float32)

    # 1. Exact parity against NumPy L1-box upper bound formula
    K_f32 = K_np[:, : n_pages * page, :].astype(np.float32).reshape(n_kv, n_pages, page, D)
    kmin_ref = K_f32.min(axis=2)
    kmax_ref = K_f32.max(axis=2)
    q_grp = q_np.reshape(n_kv, 3, D)
    u_ref = np.maximum(q_grp[:, :, None, :] * kmin_ref[:, None, :, :], q_grp[:, :, None, :] * kmax_ref[:, None, :, :]).sum(axis=-1)
    u_ref_flat = u_ref.reshape(n_q, n_pages)
    assert float(np.max(np.abs(h_np - u_ref_flat))) < 2e-3
    assert float(np.max(np.abs(kv_np - u_ref.max(axis=1)))) < 2e-3

    # 2. Mathematical upper-bound theorem: U(q, p) >= max_{t in p} (q . k_t) for EVERY head & page!
    dots = np.einsum("gqd,gpld->gqpl", q_grp, K_f32)
    max_exact = dots.max(axis=-1).reshape(n_q, n_pages)
    assert np.all(h_np + 2e-3 >= max_exact), f"Upper bound violated: min slack={float((h_np - max_exact).min())}"

    # 3. Needle-in-a-haystack retrieval & attention accuracy
    selected = idx.select(q, budget=4, n_local=4, n_sink=1)
    assert needle_page in selected
    assert 0 in selected
    assert (n_pages - 1) in selected

    dense_out = quest.quest_attention_reference(q, K, V, pages=None, page=page)
    sparse_out = quest.quest_attention_reference(q, K, V, pages=selected, page=page)
    mx.eval(dense_out, sparse_out)
    # On the needle query head (head 9), sparse attention should closely match dense attention
    err_needle = float(np.max(np.abs(np.asarray(dense_out[9]) - np.asarray(sparse_out[9]))))
    assert err_needle < 5e-2, f"Needle head sparse error too large: {err_needle}"


# --------------------------------------------------------------------------- v0.5.3 Tests


def test_v053_decode_swiglu_only_f16_and_nv_skip_rht1152_bit_exact():
    from inkling_mlx_vision.moe_v4 import (
        fast_nv_skip_swiglu_rht1152_f16,
        fast_swiglu_only_f16,
    )

    mx.random.seed(2026)
    for n_slots in (8, 48, 64):
        y_gu = mx.random.normal((n_slots, 2304)).astype(mx.float32) * 3.0
        signs = mx.where(mx.random.uniform(shape=(1152,)) > 0.5, 1.0, -1.0).astype(mx.float32)
        eids = mx.array([((i * 7) % 64) if (i % 11 != 5) else 0xFFFFFFFF for i in range(n_slots)], dtype=mx.uint32)
        exp_fid = mx.array([7 if (e % 4 != 0) else 2 for e in range(64)], dtype=mx.uint32)
        mx.eval(y_gu, signs, eids, exp_fid)

        # 1. Pure-NVFP4 fast_swiglu_only_f16 (with and without sentinel skipping)
        ref_rot0, ref_raw0 = accel._ORIG_SWIGLU_RHT1152(y_gu, signs, 4.0, 25.0, out_dtype=mx.float16, also_raw=True)
        raw0 = fast_swiglu_only_f16(y_gu, 4.0, 25.0)
        mx.eval(ref_raw0, raw0)
        assert float(mx.max(mx.abs(raw0 - ref_raw0)).item()) == 0.0

        ref_rot1, ref_raw1 = accel._ORIG_SWIGLU_RHT1152(
            y_gu, signs, 4.0, 25.0, out_dtype=mx.float16, eids_flat=eids, also_raw=True
        )
        raw1 = fast_swiglu_only_f16(y_gu, 4.0, 25.0, eids_flat=eids)
        mx.eval(ref_rot1, ref_raw1, raw1)
        assert float(mx.max(mx.abs(raw1 - ref_raw1)).item()) == 0.0

        # 2. Mixed-NVFP4 fast_nv_skip_swiglu_rht1152_f16: raw matches everywhere, rot matches on non-NVFP4 live slots
        rot2, raw2 = fast_nv_skip_swiglu_rht1152_f16(y_gu, signs, eids, exp_fid, 4.0, 25.0)
        mx.eval(rot2, raw2)
        assert float(mx.max(mx.abs(raw2 - ref_raw1)).item()) == 0.0
        eids_np = np.asarray(eids)
        fids_np = np.asarray(exp_fid)
        for s in range(n_slots):
            eid = int(eids_np[s])
            if eid != 0xFFFFFFFF and int(fids_np[eid]) != 7:
                assert float(mx.max(mx.abs(rot2[s] - ref_rot1[s])).item()) == 0.0


def test_v053_temp_calibrated_draft_sampler_unbiased_and_higher_acceptance():
    assert abs(accel.calibrate_draft_temp(0.6, scale=0.75) - 0.45) < 1e-9
    assert accel.calibrate_draft_temp(0.0, scale=0.75) == 0.0
    assert abs(accel.calibrate_draft_temp(0.6, scale=1.0) - 0.6) < 1e-9

    # Target logits vs slightly noisier 3-layer drafter logits at target temp = 0.6
    rng = np.random.default_rng(42)
    target_logits = np.array([3.2, 2.4, 1.5, 0.8, 0.1, -0.5, -1.2, -2.0], dtype=np.float64)
    # 3-layer drafter has flatter scale (~0.75x) before temperature scaling
    draft_logits = target_logits * 0.75 + rng.normal(0.0, 0.08, size=target_logits.shape)

    def softmax_t(logits, t):
        z = (logits - logits.max()) / t
        e = np.exp(z)
        return e / e.sum()

    p_target = softmax_t(target_logits, 0.6)
    q_unscaled = softmax_t(draft_logits, 0.6)
    q_calibrated = softmax_t(draft_logits, accel.calibrate_draft_temp(0.6, scale=0.75))

    # Acceptance rate alpha = sum_x min(p(x), q(x)) is strictly higher with temperature calibration
    alpha_unscaled = float(np.minimum(p_target, q_unscaled).sum())
    alpha_calibrated = float(np.minimum(p_target, q_calibrated).sum())
    assert alpha_calibrated > alpha_unscaled

    # Exact marginal output distribution under speculative rejection sampling equals p_target (0.0 bias!)
    residual = np.maximum(0.0, p_target - q_calibrated)
    residual /= residual.sum()
    p_marginal = np.minimum(p_target, q_calibrated) + (1.0 - alpha_calibrated) * residual
    assert float(np.max(np.abs(p_marginal - p_target))) < 1e-15


def test_v053_pre_tool_gen_defers_global_kv_zero_cow():
    from inkling_mlx_vision import cache as _cache_mod

    class _DummyFastCache:
        _fast = True
        _spec_recording = False

        def __init__(self):
            self.seq_len = 2500
            self.keys = [None, None]
            self.values = [None, None]
            self.kv_is_sliding = [True, False]
            self.kv_f16 = [mx.zeros((1, 4, 1024, 64), dtype=mx.float16), None]
            self.kv_u8 = [None, mx.zeros((1, 8, 8192, 256), dtype=mx.uint8)]
            self.kv_q4_sc = [None, None]
            self.kv_len = [1024, 2500]
            self.kv_start = [1476, 0]
            self.kv_cap = [1024, 8192]
            self.conv_states = [[mx.zeros((1, 3, 64), dtype=mx.float16)] * 4 for _ in range(2)]
            self._spec_base_kv_len = [1024, 2500]
            self._reserve_cap = 0

        @property
        def state(self):
            return [a for a in (*self.kv_f16, *self.kv_u8) if a is not None]

    c = _DummyFastCache()
    mx.eval(*c.state)

    # Deferred global clone stores None for non-sliding slot 1 -> c.kv_u8[1] keeps refcount == 1!
    stored = _cache_mod._clone_cache_for_store(c, share_global=True, defer_global=True)
    assert stored.kv_u8[1] is None
    assert stored.kv_f16[0] is not None

    mem_before = int(mx.get_active_memory())
    c.kv_u8[1][:, :, 2500:2501, :] = 7
    mx.eval(c.kv_u8[1])
    mem_after = int(mx.get_active_memory())
    # Zero Copy-on-Write duplication of the 16.8 MB global kv_u8 buffer!
    assert (mem_after - mem_before) < 1_000_000

    # Syncing attaches the finished global buffer zero-copy
    _cache_mod._sync_shared_global_buffers(stored, None, c, None)
    assert stored.kv_u8[1] is c.kv_u8[1]
    assert int(stored.kv_u8[1][0, 0, 2500, 0].item()) == 7


def test_v053_long_period_repetition_loop_detection():
    from inkling_mlx_vision.parser import MonitoredOutputParser

    class _Tok:
        def decode(self, toks, **_kw):
            return "x" * len(toks)

    ids = {
        "user": 200000, "model": 200001, "system": 200002, "tool": 200003,
        "text": 200004, "thinking": 200005, "eos": 200006, "end": 200010,
        "tool_json": 200011, "tool_text": 200012,
    }
    p = MonitoredOutputParser(_Tok(), ids, tool_names=set())
    p.feed(ids["thinking"])
    # 1. 1,500 non-repeating tokens do NOT trigger false positive
    for i in range(1500):
        p.feed(100 + ((i * 97 + (i // 31) * 13) % 5000))
    assert not p.done

    # 2. A 600-token reasoning loop (> old 64/256 limit, <= 2048 new limit) triggers after 2 full periods (1,200 tokens)
    p2 = MonitoredOutputParser(_Tok(), ids, tool_names=set())
    p2.feed(ids["thinking"])
    pattern = [1000 + ((i * 37 + 11) % 900) for i in range(600)]
    for step in range(1400):
        p2.feed(pattern[step % 600])
        if p2.done:
            break
    assert p2.done, "Expected 600-token periodic thinking loop to be caught within 2.3 periods"


def test_v053_char_budget_includes_tools_and_ctx_times_5():
    msgs = [{"role": "user", "content": "a" * 300_000}]
    tools = [{"type": "function", "function": {"name": "big_tool", "description": "d" * 360_000}}]
    c_no_tools = server.message_char_count(msgs)
    c_with_tools = server.message_char_count(msgs, tools=tools)
    assert c_no_tools == 300_000
    assert c_with_tools >= 660_000
    # 131072 * 5 = 655,360 chars -> messages alone fit, messages + 360K tools overflow before BPE!
    assert not server.exceeds_char_budget(c_no_tools, 131072)
    assert server.exceeds_char_budget(c_with_tools, 131072)


# --------------------------------------------------------------------------- 9. v0.5.4 Weight-Level & Draft Shortlist Optimizations


def test_v054_extract_cyrillic_token_ids_bytelevel_and_unicode(tmp_path):
    import json

    tok_json = tmp_path / "tokenizer.json"
    tok_json.write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {
                        "hello": 0,
                        "Ġworld": 1,
                        "ÐºÐ²Ð°Ð½Ñ‚": 2,  # ByteLevel UTF-8 Cyrillic ('квант')
                        "Ñ‚Ð¾ÐºÐµÐ½": 3,  # ByteLevel UTF-8 Cyrillic ('токен')
                        "привет": 4,  # Direct Unicode Cyrillic
                    }
                },
                "added_tokens": [
                    {"id": 5, "content": "<|end_message|>"},
                    {"id": 6, "content": "ÐŸÑ€Ð¸Ð¼ÐµÑ€"},
                ],
            }
        ),
        encoding="utf-8",
    )
    cyr = accel.extract_cyrillic_token_ids(tok_json)
    assert cyr == {2, 3, 4, 6}


def test_v054_verify_sparse_spec_sample_gpu_unbiased():
    """Empirical proof that sparse shortlist verification (`draft_ord_map`) is 100% unbiased wrt target `p(x)`."""
    rng = np.random.default_rng(20260504)
    V = 48
    # Non-contiguous shortlist: prefix [0..7] + sparse Cyrillic columns [20, 25, 31, 40]
    ord_map_np = np.array([0, 1, 2, 3, 4, 5, 6, 7, 20, 25, 31, 40], dtype=np.int32)
    S = len(ord_map_np)
    draft_ord_map = mx.array(ord_map_np)
    perm = rng.permutation(V).astype(np.int32)
    order_mx = mx.array(perm)
    order_sub = order_mx[draft_ord_map]

    temp = 0.8
    logits_np = rng.standard_normal((2, V)).astype(np.float32) * 1.1
    p0_ord = _softmax_np(logits_np[0], temp)
    p0_vocab = np.zeros(V, dtype=np.float64)
    p0_vocab[perm] = p0_ord

    q_logits = logits_np[0, ord_map_np] + rng.standard_normal(S).astype(np.float32) * 0.5
    q0_probs = mx.array(_softmax_np(q_logits, temp).astype(np.float32))[None, :]  # [1, S]
    ord_logits = mx.array(logits_np)

    n_trials = 3500
    counts = np.zeros(V, dtype=np.int64)
    sparse_accepts = 0
    for _ in range(n_trials):
        short_idx = spec._sample_from_probs(q0_probs[0])
        k_match_mx, bonus_tok_mx = accel.verify_sparse_spec_sample_gpu(
            ord_logits,
            order_mx,
            short_idx.reshape(1),
            q0_probs,
            cap=None,
            temp=temp,
            draft_ord_map=draft_ord_map,
        )
        mx.eval(short_idx, k_match_mx, bonus_tok_mx)
        s_i = int(short_idx.item())
        k_m = int(k_match_mx.item())
        if k_m == 1:
            tok = int(order_sub[s_i].item())
            if s_i >= 8:
                sparse_accepts += 1
        else:
            tok = int(bonus_tok_mx.item())
        counts[tok] += 1

    emp = counts / n_trials
    max_err = float(np.max(np.abs(emp - p0_vocab)))
    assert max_err < 0.028, f"Sparse shortlist verify bias too large: max_err={max_err:.4f}"
    assert sparse_accepts > 25, "Expected sparse Cyrillic rows (index >= 8) to be drafted and accepted"


def test_v054_resolve_decode_moe_routing_prefill_vs_decode_and_postloop(monkeypatch):
    class _Cfg:
        moe_k_max = 8
        moe_k_min = 8
        moe_ratio_thr = 0.0
        moe_renorm = False

    cfg = _Cfg()
    # 1. Default: static k=8 on both prefill and decode
    assert accel.resolve_decode_moe_routing(cfg, cache_slot=10, T=6) == (False, 8, 8, 0.0, False)
    assert accel.resolve_decode_moe_routing(cfg, cache_slot=10, T=64) == (False, 8, 8, 0.0, False)

    # 2. Even if config or decode env enables dynamic MoE, prefill (T > 8) stays at lossless static k=8!
    monkeypatch.setenv("LAMQ_DECODE_MOE_K_MAX", "7")
    monkeypatch.setenv("LAMQ_POSTLOOP_MOE_K_MAX", "6")
    assert accel.resolve_decode_moe_routing(cfg, cache_slot=10, T=64) == (False, 8, 8, 0.0, False)
    assert accel.resolve_decode_moe_routing(cfg, cache_slot=10, T=6) == (True, 7, 7, 0.0, True)
    # Post-loop L18..23 (cache_slot 42..47) applies LAMQ_POSTLOOP_MOE_K_MAX=6
    assert accel.resolve_decode_moe_routing(cfg, cache_slot=42, T=6) == (True, 6, 6, 0.0, True)


def test_v054_build_hybrid_quant_dir_symlinks_and_luts_merge(tmp_path):
    hi = tmp_path / "12.8GB"
    lo = tmp_path / "8.9GB"
    out = tmp_path / "hybrid"
    hi.mkdir()
    lo.mkdir()

    (hi / "config.json").write_text('{"hidden_size": 2304}', encoding="utf-8")
    (lo / "config.json").write_text('{"hidden_size": 2304}', encoding="utf-8")
    (hi / "steering.safetensors").write_bytes(b"STEER")
    np.savez(hi / "luts.npz", nvfp4=np.ones((4,), dtype=np.float32), rtr4=np.ones((8,), dtype=np.float32))
    np.savez(lo / "luts.npz", rtr1=np.zeros((2,), dtype=np.float32), rtr2=np.zeros((4,), dtype=np.float32))

    for idx in range(24):
        (hi / f"layer_{idx:02d}_meta.json").write_text('{"src": "hi"}', encoding="utf-8")
        (lo / f"layer_{idx:02d}_meta.json").write_text('{"src": "lo"}', encoding="utf-8")
        (hi / f"layer_{idx:02d}_packed.npz").write_bytes(b"H" * 1000)
        (lo / f"layer_{idx:02d}_packed.npz").write_bytes(b"L" * 400)

    manifest = accel.build_hybrid_quant_dir(hi, lo, out, low_layers=(18, 19, 20, 21, 22, 23))
    assert manifest["low_layers"] == [18, 19, 20, 21, 22, 23]
    assert manifest["saved_bytes"] == 6 * 600
    assert set(manifest["luts_keys"]) == {"nvfp4", "rtr1", "rtr2", "rtr4"}
    assert (out / "layer_17_packed.npz").resolve() == (hi / "layer_17_packed.npz").resolve()
    assert (out / "layer_18_packed.npz").resolve() == (lo / "layer_18_packed.npz").resolve()
    with np.load(out / "luts.npz") as merged:
        assert set(merged.files) == {"nvfp4", "rtr1", "rtr2", "rtr4"}


def test_v055_select_adaptive_draft_block_4tier_and_long_context():
    # Normal context (< 48K): 4 tiers {3, 4, 6, 8}
    assert spec.select_adaptive_draft_block(1.20, ctx_len=4096, max_block=8) == 3
    assert spec.select_adaptive_draft_block(1.85, ctx_len=4096, max_block=8) == 4
    assert spec.select_adaptive_draft_block(2.80, ctx_len=4096, max_block=8) == 6
    assert spec.select_adaptive_draft_block(4.20, ctx_len=4096, max_block=8) == 8

    # Long context (>= 48K): shifts down one tier {3, 4, 6} to bound multi-query SDPA KV bandwidth
    assert spec.select_adaptive_draft_block(1.85, ctx_len=65536, max_block=8) == 3
    assert spec.select_adaptive_draft_block(2.80, ctx_len=65536, max_block=8) == 4
    assert spec.select_adaptive_draft_block(4.20, ctx_len=65536, max_block=8) == 6



