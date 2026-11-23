"""Quest-style query-aware sparse KV page selection (ICML 2024) for ultra-long contexts (32K-128K).

For each 64-token KV page `p` we keep per-channel key bounds `K_min[p, d]`, `K_max[p, d]`. For any query `q`
the page's best possible dot product is bounded exactly by

    U(q, p) = sum_d max(q_d * K_min[p, d], q_d * K_max[p, d]) >= max_{t in p} q . k_t

so pages with low `U` provably cannot contain a high-attention key. `QuestPageIndex.select` returns
sink + local + top-`budget` pages; attention over that subset is an approximation (lossy), therefore this
module is a standalone building block and is NOT enabled on the default exact decode path.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np


@lru_cache(maxsize=8)
def make_quest_page_score_kernel(group: int = 3, head_dim: int = 128):
    """One simdgroup per (kv_head, page): lane handles `head_dim / 32` channels, `simd_sum` per query head.

    Outputs `head_scores[n_q, n_pages]` and `kv_scores[n_kv, n_pages]` (max over the GQA group).
    """
    per_lane = head_dim // 32
    src = f"""
    const uint lane = thread_position_in_threadgroup.x;   // 0..31
    const uint page = threadgroup_position_in_grid.y;
    const uint kvh = threadgroup_position_in_grid.z;
    const uint n_pages = meta[0];
    const ulong base = ((ulong)kvh * n_pages + page) * {head_dim}u;
    float lo[{per_lane}], hi[{per_lane}];
    #pragma unroll
    for (uint i = 0; i < {per_lane}u; i++) {{
        const uint d = lane + 32u * i;
        lo[i] = float(kmin[base + d]);
        hi[i] = float(kmax[base + d]);
    }}
    float best = -INFINITY;
    #pragma unroll
    for (uint gq = 0; gq < {group}u; gq++) {{
        const uint qh = kvh * {group}u + gq;
        float acc = 0.0f;
        #pragma unroll
        for (uint i = 0; i < {per_lane}u; i++) {{
            const float qd = q[(ulong)qh * {head_dim}u + lane + 32u * i];
            acc += metal::max(qd * lo[i], qd * hi[i]);
        }}
        acc = simd_sum(acc);
        if (lane == 0u) head_scores[(ulong)qh * n_pages + page] = acc;
        best = metal::max(best, acc);
    }}
    if (lane == 0u) kv_scores[(ulong)kvh * n_pages + page] = best;
    """
    return mx.fast.metal_kernel(
        name=f"ink_quest_page_score_g{group}_d{head_dim}",
        input_names=["q", "kmin", "kmax", "meta"],
        output_names=["head_scores", "kv_scores"],
        source=src,
    )


def quest_page_scores(q: mx.array, kmin: mx.array, kmax: mx.array) -> tuple[mx.array, mx.array]:
    """`q [n_q, D]`, `kmin/kmax [n_kv, P, D]` -> `(head_scores [n_q, P], kv_scores [n_kv, P])` (float32)."""
    n_q, D = int(q.shape[0]), int(q.shape[1])
    n_kv, P = int(kmin.shape[0]), int(kmin.shape[1])
    group = n_q // n_kv
    outs = make_quest_page_score_kernel(group, D)(
        inputs=[q.astype(mx.float32).reshape(-1), kmin.reshape(-1), kmax.reshape(-1), mx.array([P], dtype=mx.uint32)],
        grid=(32, P, n_kv),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_q, P), (n_kv, P)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return outs[0], outs[1]


class QuestPageIndex:
    """Incremental per-page key bounding boxes for one global attention layer (`K [n_kv, L, D]`)."""

    def __init__(self, page: int = 64, dtype=mx.float16):
        self.page = int(page)
        self.dtype = dtype
        self.kmin: mx.array | None = None  # [n_kv, P, D]
        self.kmax: mx.array | None = None
        self._tail: mx.array | None = None  # [n_kv, <page, D] incomplete page
        self.n_tokens = 0

    @property
    def n_pages(self) -> int:
        return 0 if self.kmin is None else int(self.kmin.shape[1])

    def _pages_from(self, K: mx.array) -> tuple[mx.array, mx.array]:
        n_kv, L, D = (int(s) for s in K.shape)
        P = L // self.page
        Kp = K[:, : P * self.page, :].reshape(n_kv, P, self.page, D)
        return mx.min(Kp, axis=2).astype(self.dtype), mx.max(Kp, axis=2).astype(self.dtype)

    def append(self, K_new: mx.array) -> None:
        """Append new keys `[n_kv, T, D]`; complete pages are summarized, the remainder is buffered."""
        K = K_new if self._tail is None else mx.concatenate([self._tail, K_new], axis=1)
        self.n_tokens += int(K_new.shape[1])
        L = int(K.shape[1])
        P = L // self.page
        if P > 0:
            lo, hi = self._pages_from(K[:, : P * self.page, :])
            self.kmin = lo if self.kmin is None else mx.concatenate([self.kmin, lo], axis=1)
            self.kmax = hi if self.kmax is None else mx.concatenate([self.kmax, hi], axis=1)
        rem = L - P * self.page
        self._tail = K[:, P * self.page :, :] if rem > 0 else None

    def build(self, K: mx.array) -> "QuestPageIndex":
        self.kmin = self.kmax = self._tail = None
        self.n_tokens = 0
        self.append(K)
        return self

    def scores(self, q: mx.array) -> tuple[mx.array, mx.array]:
        if self.kmin is None:
            raise ValueError("QuestPageIndex is empty")
        return quest_page_scores(q, self.kmin, self.kmax)

    def select(self, q: mx.array, budget: int = 64, n_local: int = 16, n_sink: int = 1) -> list[int]:
        """Union over KV heads of sink pages + last `n_local` pages + top-`budget` pages by upper bound."""
        P = self.n_pages
        if P <= budget + n_local + n_sink:
            return list(range(P))
        _, kv_sc = self.scores(q)
        mid_lo, mid_hi = n_sink, max(n_sink, P - n_local)
        k = max(0, min(budget, mid_hi - mid_lo))
        sel: set[int] = set(range(n_sink))
        sel.update(range(mid_hi, P))
        if k > 0:
            mid = kv_sc[:, mid_lo:mid_hi]
            top = mx.argpartition(-mid, k - 1, axis=-1)[:, :k] + mid_lo
            sel.update(np.array(top).reshape(-1).tolist())
        return sorted(sel)


def quest_attention_reference(q: mx.array, K: mx.array, V: mx.array, pages: list[int] | None, page: int = 64) -> mx.array:
    """Reference GQA softmax attention (`q [n_q, D]`, `K/V [n_kv, L, D]`) restricted to `pages` (None = dense)."""
    n_q, D = int(q.shape[0]), int(q.shape[1])
    n_kv, L = int(K.shape[0]), int(K.shape[1])
    if pages is not None:
        idx = [t for p in pages for t in range(p * page, min((p + 1) * page, L))]
        tail_start = (L // page) * page
        idx += [t for t in range(tail_start, L) if t not in set(idx)]
        sel = mx.array(np.array(sorted(set(idx)), dtype=np.int32))
        K, V = K[:, sel, :], V[:, sel, :]
    g = n_q // n_kv
    Kq = mx.repeat(K.astype(mx.float32), g, axis=0)
    Vq = mx.repeat(V.astype(mx.float32), g, axis=0)
    s = mx.sum(Kq * q.astype(mx.float32)[:, None, :], axis=-1) / (D ** 0.5)
    w = mx.softmax(s, axis=-1)
    return mx.sum(w[:, :, None] * Vq, axis=1)
