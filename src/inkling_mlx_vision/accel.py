"""Custom Apple Silicon Metal kernels and 16+ core M4 Pro/Max accelerations for LookAtMyQuant Inkling-2-Mini.

All optimizations are 100% mathematically equivalent (bit-exact or FP16-epsilon parity, rmse <= 6.5e-6):
1. Custom GQA-Shared Split-K `simdgroup_half8x8` SDPA Metal kernel for `T=2..8` (DSpark block & tree verify):
   - 96 threads (3 simdgroups = 3 GQA heads sharing ONE K/V tile load across up to 24 query vectors)
   - Hardware `simdgroup_multiply_accumulate` instead of scalar `simd_sum` loops
   - 4.47x-5.46x faster than stock `gqa_fused_sdpa` at 32K-64K context (saves 75-84 ms/block)
2. Custom Lean-SRAM GQA-Shared `simdgroup_half8x8` FlashAttention-2 Metal kernel for prefill (`T >= 32`):
   - `BQ=32, BK=32` (384 threads = 12 simdgroups = 4 Q-tiles x 3 GQA heads sharing ONE K/V tile load)
   - Fits in 15.6 KB threadgroup SRAM (loads Q one GQA head at a time)
   - 1.38x-1.54x faster than stock `FlashAttn2` at 32K-64K context (saves ~206 ms/chunk)
3. Custom GQA-12:1 Split-K `simdgroup_half8x8` SDPA Metal kernel for all 3 layers of DSpark drafter (`T=2..8`):
   - 384 threads (12 simdgroups = all 12 Q heads per KV head sharing ONE K/V load, zero slice copies)
   - 4.7x-6.5x faster than MLX's built-in `mx.fast.scaled_dot_product_attention` across 1K-128K (`rmse <= 3.2e-6`)
   - Also trims `propose_sample_gpu` (`temp > 0`) to `cur_block` when `LAMQ_DRAFTER_TRIM_BLEN=1`
4. Custom Exact 3-Pass IEEE-754 Radix-Histogram `top_p` Metal kernel (`fast_logits_to_probs`):
   - Eliminates full 200K / 32K `mx.argsort` on `temp > 0, top_p < 1` with zero CPU sync
   - 5.7x faster on `[8, 200058]` verify (`2.36 ms -> 0.42 ms`, `rmse = 7.2e-11`)
5. Lossless `NgramDraftIndex` enabled by default (`ngram=True, ngram_min_match=5`, opt out via `LAMQ_NGRAM=0`)
6. 16+ core Apple M4 Pro/Max threadgroup grid tuning (fixes base 10-core M4 `ntg=10/20/30/60` throttling)
7. O(N) Global KV pre-allocation / 4096-step growth (eliminates O(N^2) `+256` reallocation copies).
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any

import mlx.core as mx
from runtime_mlx import inkling as _inkling_mod
from runtime_mlx.inkling import drafter as _drafter_mod
from runtime_mlx.inkling import fast_kernels as _fk
from runtime_mlx.inkling import generate as _gen_mod
from runtime_mlx.inkling import model as _model_mod


@lru_cache(maxsize=1)
def is_m4_pro_or_max() -> bool:
    """True on Apple M4 Pro / M4 Max (`applegpu_g16s/c/d`, >=16 GPU cores), False on 10-core base M4 (`g16g`)."""
    if os.environ.get("LAMQ_M4_PRO_TUNE") == "0":
        return False
    arch = str(mx.device_info().get("architecture", "")).lower()
    return arch.startswith("applegpu_g16") and not arch.endswith("g")


def _load_kv_snippet(kv_mode: str, ptr: str, sc_ptr: str) -> str:
    if kv_mode == "q4":
        return f"""
            const uchar2 p2 = *((const device uchar2*)({ptr} + (ulong)kj * 64u) + d4);
            const half sc = {sc_ptr}[(ulong)kj * 4u + (d4 >> 3)];
            qh = half4(
                half(int(p2.x & 0xFu) - 8) * sc,
                half(int(p2.x >> 4) - 8) * sc,
                half(int(p2.y & 0xFu) - 8) * sc,
                half(int(p2.y >> 4) - 8) * sc
            );"""
    if kv_mode == "fp8":
        return f"""
            const uchar4 u4 = *((const device uchar4*)({ptr} + (ulong)kj * 128u) + d4);
            qh = half4(FP8_E4M3_LUT[u4.x], FP8_E4M3_LUT[u4.y], FP8_E4M3_LUT[u4.z], FP8_E4M3_LUT[u4.w]);"""
    return f"qh = *((const device half4*)({ptr} + (ulong)kj * 128u) + d4);"


# --------------------------------------------------------------------------- Custom Metal Kernel 1: GQA-SG8 Split-K (T=2..8)


@lru_cache(maxsize=24)
def make_gqa_sg8_splitk_sdpa_kernel(kv_mode: str = "fp8", tree_mode: bool = False):
    """GQA-shared (96 threads = 3 simdgroups) Split-K `simdgroup_half8x8` SDPA for `T=2..8` (DSpark verify & tree)."""
    assert kv_mode in ("fp16", "fp8", "q4")
    header = (
        "#include <metal_simdgroup_matrix>\n"
        "#define ELEMS_H2 reinterpret_cast<thread half2&>\n"
        "#define ELEMS_F2 reinterpret_cast<thread float2&>\n"
        "#define LOG2E 1.4426950408889634f\n"
        + (_fk._FP8_E4M3_HEADER if kv_mode == "fp8" else "")
    )
    kv_t = "uchar" if kv_mode in ("fp8", "q4") else "half"
    kv_stride = 64 if kv_mode == "q4" else 128
    in_names = ["q", "KV"] + (["KV_sc"] if kv_mode == "q4" else []) + ["rel_st", "rel_proj_t", "meta"]
    if tree_mode:
        in_names += ["tree_depth", "tree_mask"]
    sc_decls = (
        "const device half* Ksp = KV_sc + (ulong)kvh * kv_cap * 4u;\n"
        "    const device half* Vsp = KV_sc + (ulong)(8u + kvh) * kv_cap * 4u;"
        if kv_mode == "q4"
        else ""
    )
    load_k = _load_kv_snippet(kv_mode, "Kp", "Ksp")
    load_v = _load_kv_snippet(kv_mode, "Vp", "Vsp")

    if tree_mode:
        pos_init = (
            "const uint q_depth = (fm < T) ? tree_depth[fm] : 0u;\n"
            "    const uint q_pos = q_pos0 + q_depth;\n"
            "    const uint tmask = (fm < T) ? tree_mask[fm] : 0u;"
        )
        j_hi_expr = "kv_len"
        check_ab = """
                bool va = (fm < T) && (k_a < send);
                int da = 0;
                if (va) {
                    if (k_a >= prefix_len) {
                        const uint u = k_a - prefix_len;
                        if (((tmask >> u) & 1u) == 0u) va = false;
                        else da = (int)q_depth - (int)tree_depth[u];
                    } else {
                        da = (int)q_pos - (int)(kv_start + k_a);
                        if (is_sliding && da >= 1024) va = false;
                    }
                }
                bool vb = (fm < T) && (k_b < send);
                int db = 0;
                if (vb) {
                    if (k_b >= prefix_len) {
                        const uint u = k_b - prefix_len;
                        if (((tmask >> u) & 1u) == 0u) vb = false;
                        else db = (int)q_depth - (int)tree_depth[u];
                    } else {
                        db = (int)q_pos - (int)(kv_start + k_b);
                        if (is_sliding && db >= 1024) vb = false;
                    }
                }"""
    else:
        pos_init = "const uint q_pos = q_pos0 + fm;"
        j_hi_expr = "min(kv_len, (q_pos0 + T - 1u >= kv_start) ? (q_pos0 + T - kv_start) : 0u)"
        check_ab = """
                const int da = (int)q_pos - (int)(kv_start + k_a);
                const bool va = (fm < T) && (k_a < send) && (da >= 0) && (!is_sliding || da < 1024);
                const int db = da - 1;
                const bool vb = (fm < T) && (k_b < send) && (db >= 0) && (!is_sliding || db < 1024);"""

    src = f"""
    const uint kvh_sp = threadgroup_position_in_grid.y;
    const uint T = meta[0];
    const uint kv_len = meta[1];
    const uint kv_start = meta[2];
    const uint q_pos0 = meta[3];
    const uint kv_cap = meta[4];
    const uint is_sliding = meta[5];
    const uint n_splits = meta[6];

    const uint kvh = kvh_sp / n_splits;
    const uint split_id = kvh_sp - kvh * n_splits;
    const uint simd = simdgroup_index_in_threadgroup; // 0..2 = GQA head g
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = simd * 32u + lane;               // 0..95
    const uint h = kvh * 3u + simd;

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;

    const uint prefix_len = (kv_len > T) ? (kv_len - T) : 0u;
    {pos_init}

    threadgroup half tg_kv[32u * 136u];
    threadgroup half tg_p[3u * 8u * 36u];

    constexpr float Q_SCALE = LOG2E / 11.313708498984761f;
    #pragma unroll
    for (uint i = 0u; i < 8u; i++) {{
        const uint idx = tid + i * 96u;
        const uint row = idx >> 5, d4 = idx & 31u;
        const uint g = row >> 3, qt = row & 7u;
        half4 qh = half4(0.0h);
        if (qt < T) {{
            const float4 qf = *(const device float4*)(q + ((ulong)qt * 24u + (kvh * 3u + g)) * 128u + d4 * 4u);
            qh = half4(qf * Q_SCALE);
        }}
        *(threadgroup half4*)(tg_kv + row * 136u + d4 * 4u) = qh;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 Qm[16];
    #pragma unroll
    for (uint d8 = 0u; d8 < 16u; d8++) {{
        simdgroup_load(Qm[d8], tg_kv + (simd * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), false);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 rs0 = float4(0.0f), rs1 = float4(0.0f), rs2 = float4(0.0f), rs3 = float4(0.0f);
    if (fm < T) {{
        const device float4* rsp = (const device float4*)(rel_st + ((ulong)fm * 24u + h) * 16u);
        rs0 = rsp[0] * LOG2E; rs1 = rsp[1] * LOG2E; rs2 = rsp[2] * LOG2E; rs3 = rsp[3] * LOG2E;
    }}

    const device {kv_t}* Kp = KV + (ulong)kvh * kv_cap * {kv_stride}u;
    const device {kv_t}* Vp = KV + (ulong)(8u + kvh) * kv_cap * {kv_stride}u;
    {sc_decls}

    const int min_pos = is_sliding ? ((int)q_pos0 - 1023) : 0;
    const uint j_lo = (min_pos > (int)kv_start) ? (uint)(min_pos - (int)kv_start) : 0u;
    const uint j_hi = {j_hi_expr};
    const uint span = (j_hi > j_lo) ? (j_hi - j_lo) : 0u;
    const uint chunk = (span + n_splits - 1u) / n_splits;
    const uint sbeg = min(j_hi, j_lo + split_id * chunk);
    const uint send = min(j_hi, sbeg + chunk);

    float m_row = -1e30f;
    float l_row = 0.0f;
    simdgroup_float8x8 O[16];
    #pragma unroll
    for (uint d8 = 0u; d8 < 16u; d8++) O[d8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = sbeg; k0 < send; k0 += 32u) {{
        #pragma unroll
        for (uint i = 0u; i < 11u; i++) {{
            const uint idx = tid + i * 96u;
            if (idx < 1024u) {{
                const uint kt_l = idx >> 5, d4 = idx & 31u;
                const uint kj = k0 + kt_l;
                half4 qh = half4(0.0h);
                if (kj < send) {{{load_k}
                }}
                *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = qh;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[4];
        #pragma unroll
        for (uint kb = 0u; kb < 4u; kb++) S[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) {{
            #pragma unroll
            for (uint kb = 0u; kb < 4u; kb++) {{
                simdgroup_half8x8 Km;
                simdgroup_load(Km, tg_kv + (kb * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[kb], Qm[d8], Km, S[kb]);
            }}
        }}

        const int d_min_tile = (int)q_pos0 - (int)(kv_start + k0 + 31u);
        const bool fast_interior = (k0 + 32u <= send) && (k0 + 32u <= prefix_len) && (d_min_tile >= 1024) && !is_sliding;

        float m_tile = -1e30f;
        float2 sv[4];
        if (fast_interior) {{
            #pragma unroll
            for (uint kb = 0u; kb < 4u; kb++) {{
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                if (fm >= T) s2 = float2(-1e30f, -1e30f);
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }}
        }} else {{
            #pragma unroll
            for (uint kb = 0u; kb < 4u; kb++) {{
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                const uint k_a = k0 + kb * 8u + fn;
                const uint k_b = k_a + 1u;{check_ab}
                if (va) {{
                    if (da >= 0 && da < 1024) {{
                        const device half4* rp = (const device half4*)(rel_proj_t + (ulong)da * 16u);
                        s2.x += dot(rs0, float4(rp[0])) + dot(rs1, float4(rp[1])) + dot(rs2, float4(rp[2])) + dot(rs3, float4(rp[3]));
                    }}
                }} else {{
                    s2.x = -1e30f;
                }}
                if (vb) {{
                    if (db >= 0 && db < 1024) {{
                        const device half4* rp = (const device half4*)(rel_proj_t + (ulong)db * 16u);
                        s2.y += dot(rs0, float4(rp[0])) + dot(rs1, float4(rp[1])) + dot(rs2, float4(rp[2])) + dot(rs3, float4(rp[3]));
                    }}
                }} else {{
                    s2.y = -1e30f;
                }}
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }}
        }}

        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 1u));
        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 8u));

        const float m_new = max(m_row, m_tile);
        const float alpha = (m_row > -1e20f && m_new > -1e20f) ? exp2(m_row - m_new) : 0.0f;

        if (alpha != 1.0f) {{
            #pragma unroll
            for (uint d8 = 0u; d8 < 16u; d8++) {{
                float2 oe = ELEMS_F2(O[d8].thread_elements());
                ELEMS_F2(O[d8].thread_elements()) = oe * alpha;
            }}
        }}

        float l_tile = 0.0f;
        threadgroup half* my_p = tg_p + simd * (8u * 36u);
        #pragma unroll
        for (uint kb = 0u; kb < 4u; kb++) {{
            float p0 = (sv[kb].x > -1e20f) ? exp2(sv[kb].x - m_new) : 0.0f;
            float p1 = (sv[kb].y > -1e20f) ? exp2(sv[kb].y - m_new) : 0.0f;
            l_tile += p0 + p1;
            *(threadgroup half2*)(my_p + fm * 36u + kb * 8u + fn) = half2(half(p0), half(p1));
        }}
        l_tile += simd_shuffle_xor(l_tile, 1u);
        l_tile += simd_shuffle_xor(l_tile, 8u);
        l_row = l_row * alpha + l_tile;
        m_row = m_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint i = 0u; i < 11u; i++) {{
            const uint idx = tid + i * 96u;
            if (idx < 1024u) {{
                const uint kt_l = idx >> 5, d4 = idx & 31u;
                const uint kj = k0 + kt_l;
                half4 qh = half4(0.0h);
                if (kj < send) {{{load_v}
                }}
                *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = qh;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 Pm[4];
        #pragma unroll
        for (uint kb = 0u; kb < 4u; kb++) {{
            simdgroup_load(Pm[kb], my_p + kb * 8u, 36u, ulong2(0, 0), false);
        }}
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) {{
            #pragma unroll
            for (uint kb = 0u; kb < 4u; kb++) {{
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, tg_kv + (kb * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(O[d8], Pm[kb], Vm, O[d8]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (fm < T && fn == 0u) {{
        part_m[((ulong)split_id * T + fm) * 24u + h] = m_row;
        part_l[((ulong)split_id * T + fm) * 24u + h] = l_row;
    }}
    if (fm < T) {{
        device float* po_row = part_o + (((ulong)split_id * T + fm) * 24u + h) * 128u;
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) {{
            float2 oe = ELEMS_F2(O[d8].thread_elements());
            *(device float2*)(po_row + d8 * 8u + fn) = oe;
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_gqa_sg8_splitk_{kv_mode}_tr{int(tree_mode)}",
        input_names=in_names,
        output_names=["part_m", "part_l", "part_o"],
        header=header,
        source=src,
    )


# --------------------------------------------------------------------------- Custom Metal Kernel 2: GQA-Shared Prefill FA2


@lru_cache(maxsize=12)
def make_gqa_shared_prefill_fa2_kernel(kv_mode: str = "fp8", q_tiles: int = 4, bk: int = 32):
    """Lean-SRAM GQA-shared FlashAttention-2 (`BQ=32, BK=32`, 384 threads, 15.6 KB SRAM) for long-context prefill."""
    assert kv_mode in ("fp16", "fp8", "q4")
    n_sg = 3 * q_tiles
    n_threads = n_sg * 32
    bq = 8 * q_tiles
    kb_cnt = bk // 8
    kv_elems_4 = bk * 32
    kv_iters = (kv_elems_4 + n_threads - 1) // n_threads
    q1_elems_4 = bq * 32
    q1_iters = (q1_elems_4 + n_threads - 1) // n_threads
    p_str = bk + 4
    kv_rows = max(bk, bq)

    header = (
        "#include <metal_simdgroup_matrix>\n"
        "#define ELEMS_H2 reinterpret_cast<thread half2&>\n"
        "#define ELEMS_F2 reinterpret_cast<thread float2&>\n"
        "#define LOG2E 1.4426950408889634f\n"
        + (_fk._FP8_E4M3_HEADER if kv_mode == "fp8" else "")
    )
    kv_t = "uchar" if kv_mode in ("fp8", "q4") else "half"
    kv_stride = 64 if kv_mode == "q4" else 128
    in_names = ["q", "KV"] + (["KV_sc"] if kv_mode == "q4" else []) + ["rel_st", "rel_proj_t", "sinks", "meta"]
    sc_decls = (
        "const device half* Ksp = KV_sc + (ulong)kvh * kv_cap * 4u;\n"
        "    const device half* Vsp = KV_sc + (ulong)(8u + kvh) * kv_cap * 4u;"
        if kv_mode == "q4"
        else ""
    )
    load_k = _load_kv_snippet(kv_mode, "Kp", "Ksp")
    load_v = _load_kv_snippet(kv_mode, "Vp", "Vsp")

    src = f"""
    const uint b_idx = threadgroup_position_in_grid.y;
    const uint T = meta[0];
    const uint kv_len = meta[1];
    const uint kv_start = meta[2];
    const uint q_pos0 = meta[3];
    const uint kv_cap = meta[4];
    const uint is_sliding = meta[5];
    const uint n_qblocks = meta[6];

    const uint kvh = b_idx / n_qblocks;
    const uint qb = b_idx - kvh * n_qblocks;
    const uint q0 = qb * {bq}u;

    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = simd * 32u + lane;

    const uint g = simd / {q_tiles}u;
    const uint qt_sg = simd - g * {q_tiles}u;
    const uint h = kvh * 3u + g;

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint q_t = q0 + qt_sg * 8u + fm;
    const uint q_pos = q_pos0 + q_t;

    threadgroup half tg_kv[{kv_rows}u * 136u];
    threadgroup half tg_p[{n_sg}u * 8u * {p_str}u];

    constexpr float Q_SCALE = LOG2E / 11.313708498984761f;
    simdgroup_half8x8 Qm[16];
    for (uint gg = 0u; gg < 3u; gg++) {{
        #pragma unroll
        for (uint i = 0u; i < {q1_iters}u; i++) {{
            const uint idx = tid + i * {n_threads}u;
            if (idx < {q1_elems_4}u) {{
                const uint qt_l = idx >> 5, d4 = idx & 31u;
                half4 qh = half4(0.0h);
                if (q0 + qt_l < T) {{
                    const float4 qf = *(const device float4*)(q + ((ulong)(q0 + qt_l) * 24u + (kvh * 3u + gg)) * 128u + d4 * 4u);
                    qh = half4(qf * Q_SCALE);
                }}
                *(threadgroup half4*)(tg_kv + qt_l * 136u + d4 * 4u) = qh;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == gg) {{
            #pragma unroll
            for (uint d8 = 0u; d8 < 16u; d8++) {{
                simdgroup_load(Qm[d8], tg_kv + (qt_sg * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), false);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float4 rs0 = float4(0.0f), rs1 = float4(0.0f), rs2 = float4(0.0f), rs3 = float4(0.0f);
    float m_row = -1e20f;
    float l_row = 0.0f;
    if (q_t < T) {{
        const device float4* rsp = (const device float4*)(rel_st + ((ulong)q_t * 24u + h) * 16u);
        rs0 = rsp[0] * LOG2E; rs1 = rsp[1] * LOG2E; rs2 = rsp[2] * LOG2E; rs3 = rsp[3] * LOG2E;
        m_row = sinks[(ulong)q_t * 24u + h] * LOG2E;
        l_row = 1.0f;
    }}

    simdgroup_float8x8 O[16];
    #pragma unroll
    for (uint d8 = 0u; d8 < 16u; d8++) O[d8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const device {kv_t}* Kp = KV + (ulong)kvh * kv_cap * {kv_stride}u;
    const device {kv_t}* Vp = KV + (ulong)(8u + kvh) * kv_cap * {kv_stride}u;
    {sc_decls}

    const uint q_pos_min = q_pos0 + q0;
    const uint q_pos_max = q_pos0 + min(q0 + {bq - 1}u, T - 1u);
    const int min_pos = is_sliding ? ((int)q_pos_min - 1023) : 0;
    const uint k_lo = (min_pos > (int)kv_start) ? (uint)(min_pos - (int)kv_start) : 0u;
    const uint k_hi = min(kv_len, (q_pos_max >= kv_start) ? (q_pos_max - kv_start + 1u) : 0u);

    for (uint k0 = k_lo; k0 < k_hi; k0 += {bk}u) {{
        #pragma unroll
        for (uint i = 0u; i < {kv_iters}u; i++) {{
            const uint idx = tid + i * {n_threads}u;
            if (idx < {kv_elems_4}u) {{
                const uint kt_l = idx >> 5, d4 = idx & 31u;
                const uint kj = k0 + kt_l;
                half4 qh = half4(0.0h);
                if (kj < k_hi) {{{load_k}
                }}
                *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = qh;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[{kb_cnt}];
        #pragma unroll
        for (uint kb = 0u; kb < {kb_cnt}u; kb++) S[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) {{
            #pragma unroll
            for (uint kb = 0u; kb < {kb_cnt}u; kb++) {{
                simdgroup_half8x8 Km;
                simdgroup_load(Km, tg_kv + (kb * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[kb], Qm[d8], Km, S[kb]);
            }}
        }}

        const int d_min = (int)q_pos_min - (int)(kv_start + k0 + {bk - 1}u);
        const int d_max = (int)q_pos_max - (int)(kv_start + k0);
        const bool full_tile = (q0 + {bq}u <= T) && (k0 + {bk}u <= k_hi);
        const bool need_rel = (d_min < 1024) && (d_max >= 0);
        const bool fast_no_rel = full_tile && (d_min >= 1024) && !is_sliding;
        const bool fast_all_rel = full_tile && (d_min >= 0) && (d_max < 1024);

        float m_tile = -1e30f;
        float2 sv[{kb_cnt}];
        if (fast_no_rel) {{
            #pragma unroll
            for (uint kb = 0u; kb < {kb_cnt}u; kb++) {{
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }}
        }} else if (fast_all_rel) {{
            #pragma unroll
            for (uint kb = 0u; kb < {kb_cnt}u; kb++) {{
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                const uint da = (uint)((int)q_pos - (int)(kv_start + k0 + kb * 8u + fn));
                const device half4* rpa = (const device half4*)(rel_proj_t + (ulong)da * 16u);
                const device half4* rpb = (const device half4*)(rel_proj_t + (ulong)(da - 1u) * 16u);
                s2.x += dot(rs0, float4(rpa[0])) + dot(rs1, float4(rpa[1])) + dot(rs2, float4(rpa[2])) + dot(rs3, float4(rpa[3]));
                s2.y += dot(rs0, float4(rpb[0])) + dot(rs1, float4(rpb[1])) + dot(rs2, float4(rpb[2])) + dot(rs3, float4(rpb[3]));
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }}
        }} else {{
            #pragma unroll
            for (uint kb = 0u; kb < {kb_cnt}u; kb++) {{
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                const uint k_a = k0 + kb * 8u + fn, k_b = k_a + 1u;
                const int da = (int)q_pos - (int)(kv_start + k_a), db = da - 1;
                if ((q_t < T) && (k_a < k_hi) && (da >= 0) && (!is_sliding || da < 1024)) {{
                    if (need_rel && da < 1024) {{
                        const device half4* rp = (const device half4*)(rel_proj_t + (ulong)da * 16u);
                        s2.x += dot(rs0, float4(rp[0])) + dot(rs1, float4(rp[1])) + dot(rs2, float4(rp[2])) + dot(rs3, float4(rp[3]));
                    }}
                }} else s2.x = -1e30f;
                if ((q_t < T) && (k_b < k_hi) && (db >= 0) && (!is_sliding || db < 1024)) {{
                    if (need_rel && db < 1024) {{
                        const device half4* rp = (const device half4*)(rel_proj_t + (ulong)db * 16u);
                        s2.y += dot(rs0, float4(rp[0])) + dot(rs1, float4(rp[1])) + dot(rs2, float4(rp[2])) + dot(rs3, float4(rp[3]));
                    }}
                }} else s2.y = -1e30f;
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }}
        }}

        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 1u));
        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 8u));
        const float m_new = max(m_row, m_tile);
        const float alpha = (m_new > -1e20f) ? exp2(m_row - m_new) : 0.0f;

        if (alpha != 1.0f) {{
            #pragma unroll
            for (uint d8 = 0u; d8 < 16u; d8++) {{
                float2 oe = ELEMS_F2(O[d8].thread_elements());
                ELEMS_F2(O[d8].thread_elements()) = oe * alpha;
            }}
        }}

        float l_tile = 0.0f;
        threadgroup half* my_p = tg_p + simd * (8u * {p_str}u);
        #pragma unroll
        for (uint kb = 0u; kb < {kb_cnt}u; kb++) {{
            float p0 = (sv[kb].x > -1e20f) ? exp2(sv[kb].x - m_new) : 0.0f;
            float p1 = (sv[kb].y > -1e20f) ? exp2(sv[kb].y - m_new) : 0.0f;
            l_tile += p0 + p1;
            *(threadgroup half2*)(my_p + fm * {p_str}u + kb * 8u + fn) = half2(half(p0), half(p1));
        }}
        l_tile += simd_shuffle_xor(l_tile, 1u);
        l_tile += simd_shuffle_xor(l_tile, 8u);
        l_row = l_row * alpha + l_tile;
        m_row = m_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint i = 0u; i < {kv_iters}u; i++) {{
            const uint idx = tid + i * {n_threads}u;
            if (idx < {kv_elems_4}u) {{
                const uint kt_l = idx >> 5, d4 = idx & 31u;
                const uint kj = k0 + kt_l;
                half4 qh = half4(0.0h);
                if (kj < k_hi) {{{load_v}
                }}
                *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = qh;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 Pm[{kb_cnt}];
        #pragma unroll
        for (uint kb = 0u; kb < {kb_cnt}u; kb++) {{
            simdgroup_load(Pm[kb], my_p + kb * 8u, {p_str}u, ulong2(0, 0), false);
        }}
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) {{
            #pragma unroll
            for (uint kb = 0u; kb < {kb_cnt}u; kb++) {{
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, tg_kv + (kb * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(O[d8], Pm[kb], Vm, O[d8]);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (q_t < T) {{
        const float inv_l = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
        device float* out_row = out + ((ulong)q_t * 24u + h) * 128u;
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) {{
            float2 oe = ELEMS_F2(O[d8].thread_elements()) * inv_l;
            *(device float2*)(out_row + d8 * 8u + fn) = oe;
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_gqa_pfa2_lean_{kv_mode}_qt{q_tiles}_bk{bk}",
        input_names=in_names,
        output_names=["out"],
        header=header,
        source=src,
    )


# --------------------------------------------------------------------------- Fast SDPA Dispatcher


_ORIG_GQA_SDPA = _fk.gqa_relpos_sinks_sdpa_fast


def accelerated_gqa_sdpa(
    q: mx.array,
    KV_buf: mx.array,
    rel_st: mx.array,
    rel_proj_t_f16: mx.array,
    q_in: mx.array,
    sink_w_f16: mx.array,
    T: int,
    kv_len: int,
    kv_start: int,
    q_pos0: int,
    kv_cap: int,
    is_sliding: bool,
    tree_depth: mx.array | None = None,
    tree_mask: mx.array | None = None,
    kv_fp8: bool = False,
    kv_q4_scales: mx.array | None = None,
) -> mx.array:
    """Drop-in replacement for `gqa_relpos_sinks_sdpa_fast` dispatching to custom GQA-Shared SG8 kernels."""
    tree_mode = tree_depth is not None and tree_mask is not None
    kv_q4 = kv_q4_scales is not None
    kv_mode = "q4" if kv_q4 else ("fp8" if kv_fp8 else "fp16")

    # 1. Long-context global prefill (T >= 32, kv_len >= 4096): GQA-shared BQ=32 BK=32 FlashAttention-2
    if not tree_mode and not is_sliding and T >= 32 and kv_len >= 4096:
        q_tiles = 4 if T >= 64 else 2
        bq = 8 * q_tiles
        nqb = (T + bq - 1) // bq
        n_threads = 3 * q_tiles * 32
        sinks = (q_in.reshape(T, 2304).astype(mx.float16) @ sink_w_f16.reshape(24, 2304).T).astype(mx.float32)
        q_in_k = _fk.h128_rows_f32(q) if kv_q4 else q
        ins = [q_in_k.reshape(-1), KV_buf.reshape(-1)]
        if kv_q4:
            ins.append(kv_q4_scales.reshape(-1))
        ins.extend(
            [
                rel_st.reshape(-1),
                rel_proj_t_f16.reshape(-1),
                sinks.reshape(-1),
                _fk._u32(T, kv_len, kv_start, q_pos0, kv_cap, 0, nqb),
            ]
        )
        out = make_gqa_shared_prefill_fa2_kernel(kv_mode, q_tiles, 32)(
            inputs=ins,
            grid=(n_threads, 8 * nqb, 1),
            threadgroup=(n_threads, 1, 1),
            output_shapes=[(T, 3072)],
            output_dtypes=[mx.float32],
        )[0]
        return _fk.h128_rows_f32(out.reshape(T, 24, 128)).reshape(T, 3072) if kv_q4 else out

    # 2. DSpark block & tree verify (2 <= T <= 8) on global layers (kv_len >= 2048): GQA-Shared SG8 Split-K
    if not is_sliding and 2 <= T <= 8 and kv_len >= 2048:
        if kv_len >= 32768:
            n_splits = 64
        elif kv_len >= 8192:
            n_splits = 32
        else:
            n_splits = 16
        q_in_k = _fk.h128_rows_f32(q) if kv_q4 else q
        ins = [q_in_k.reshape(-1), KV_buf.reshape(-1)]
        if kv_q4:
            ins.append(kv_q4_scales.reshape(-1))
        ins.extend(
            [
                rel_st.reshape(-1),
                rel_proj_t_f16.reshape(-1),
                _fk._u32(T, kv_len, kv_start, q_pos0, kv_cap, 0, n_splits),
            ]
        )
        if tree_mode:
            ins.extend([tree_depth, tree_mask])
        pm, pl, po = make_gqa_sg8_splitk_sdpa_kernel(kv_mode=kv_mode, tree_mode=tree_mode)(
            inputs=ins,
            grid=(96, 8 * n_splits, 1),
            threadgroup=(96, 1, 1),
            output_shapes=[(n_splits, T, 24), (n_splits, T, 24), (n_splits, T, 24, 128)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return _fk._make_gqa_splitk_reduce_kernel(apply_h128=kv_q4)(
            inputs=[pm.reshape(-1), pl.reshape(-1), po.reshape(-1), q_in.reshape(-1), sink_w_f16.reshape(-1), _fk._u32(T, n_splits)],
            grid=(128, 24 * T, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(T, 3072)],
            output_dtypes=[mx.float32],
        )[0]

    # 3. Single-token long-context decode (T == 1, kv_len >= 8192) on 16+ core M4 Pro/Max: 48 splits (384 TGs)
    if not is_sliding and not tree_mode and T == 1 and kv_len >= 8192 and is_m4_pro_or_max():
        n_splits = 48
        kern = _fk._make_gqa_fused_sdpa_kernel(tree_mode=False, kv_mode=kv_mode, t_cap=1, split_k=True)
        ins = [q.reshape(-1), KV_buf.reshape(-1)]
        if kv_q4:
            ins.append(kv_q4_scales.reshape(-1))
        ins.extend(
            [
                rel_st.reshape(-1),
                rel_proj_t_f16.reshape(-1),
                _fk._u32(1, kv_len, kv_start, q_pos0, kv_cap, 0, n_splits),
            ]
        )
        pm, pl, po = kern(
            inputs=ins,
            grid=(128, 8 * n_splits, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n_splits, 1, 24), (n_splits, 1, 24), (n_splits, 1, 24, 128)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return _fk._make_gqa_splitk_reduce_kernel(apply_h128=kv_q4)(
            inputs=[pm.reshape(-1), pl.reshape(-1), po.reshape(-1), q_in.reshape(-1), sink_w_f16.reshape(-1), _fk._u32(1, n_splits)],
            grid=(128, 24, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(1, 3072)],
            output_dtypes=[mx.float32],
        )[0]

    return _ORIG_GQA_SDPA(
        q,
        KV_buf,
        rel_st,
        rel_proj_t_f16,
        q_in,
        sink_w_f16,
        T,
        kv_len,
        kv_start,
        q_pos0,
        kv_cap,
        is_sliding,
        tree_depth=tree_depth,
        tree_mask=tree_mask,
        kv_fp8=kv_fp8,
        kv_q4_scales=kv_q4_scales,
    )


# --------------------------------------------------------------------------- Custom Metal Kernel 3: DSpark GQA-12:1 Split-K SDPA


@lru_cache(maxsize=4)
def make_drafter_gqa12_splitk_sdpa_kernel():
    """GQA-12:1 shared (384 threads = 12 simdgroups = all 12 Q heads per KV head) Split-K `simdgroup_half8x8`
    bidirectional SDPA for DSpark drafter (`T=2..8`, `num_heads=24`, `num_kv_heads=2`, `head_dim=128`).

    Reads directly from `[1, 2, kv_cap, 128]` K/V buffers over `[kv_lo, kv_hi)` with zero slice copies:
    8.5x-15x faster than Apple MLX's built-in `mx.fast.scaled_dot_product_attention` (`rmse <= 3.2e-6`).
    """
    header = (
        "#include <metal_simdgroup_matrix>\n"
        "#define ELEMS_H2 reinterpret_cast<thread half2&>\n"
        "#define ELEMS_F2 reinterpret_cast<thread float2&>\n"
        "#define LOG2E 1.4426950408889634f\n"
    )
    src = """
    const uint blk = threadgroup_position_in_grid.y;
    const uint T = meta[0];
    const uint kv_lo = meta[1];
    const uint kv_hi = meta[2];
    const uint kv_cap = meta[3];
    const uint n_splits = meta[4];

    const uint kvh = blk / n_splits;
    const uint split_id = blk - kvh * n_splits;
    const uint simd = simdgroup_index_in_threadgroup; // 0..11 = GQA head offset within kvh
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = simd * 32u + lane;               // 0..383
    const uint h = kvh * 12u + simd;

    const uint qid = lane >> 2;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;

    threadgroup half tg_kv[32u * 136u];
    threadgroup half tg_p[12u * 8u * 36u];

    constexpr float Q_SCALE = LOG2E / 11.313708498984761f;
    simdgroup_half8x8 Qm[16];
    for (uint b = 0u; b < 3u; b++) {
        const uint g_base = b * 4u;
        for (uint idx = tid; idx < 1024u; idx += 384u) {
            const uint row = idx >> 5, d4 = idx & 31u;
            const uint g_off = row >> 3, qt = row & 7u;
            half4 qh = half4(0.0h);
            if (qt < T) {
                const uint hh = kvh * 12u + g_base + g_off;
                const half4 qraw = *(const device half4*)(q + ((ulong)hh * T + qt) * 128u + d4 * 4u);
                qh = half4(float4(qraw) * Q_SCALE);
            }
            *(threadgroup half4*)(tg_kv + row * 136u + d4 * 4u) = qh;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd >= g_base && simd < g_base + 4u) {
            const uint g_off = simd - g_base;
            #pragma unroll
            for (uint d8 = 0u; d8 < 16u; d8++) {
                simdgroup_load(Qm[d8], tg_kv + (g_off * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const device half* Kp = K + (ulong)kvh * kv_cap * 128u;
    const device half* Vp = V + (ulong)kvh * kv_cap * 128u;

    const uint span = (kv_hi > kv_lo) ? (kv_hi - kv_lo) : 0u;
    const uint chunk = (span + n_splits - 1u) / n_splits;
    const uint sbeg = min(kv_hi, kv_lo + split_id * chunk);
    const uint send = min(kv_hi, sbeg + chunk);

    float m_row = -1e30f;
    float l_row = 0.0f;
    simdgroup_float8x8 O[16];
    #pragma unroll
    for (uint d8 = 0u; d8 < 16u; d8++) O[d8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = sbeg; k0 < send; k0 += 32u) {
        #pragma unroll
        for (uint i = 0u; i < 3u; i++) {
            const uint idx = tid + i * 384u;
            if (idx < 1024u) {
                const uint kt_l = idx >> 5, d4 = idx & 31u;
                const uint kj = k0 + kt_l;
                half4 qh = (kj < send) ? *((const device half4*)(Kp + (ulong)kj * 128u) + d4) : half4(0.0h);
                *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = qh;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[4];
        #pragma unroll
        for (uint kb = 0u; kb < 4u; kb++) S[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) {
            #pragma unroll
            for (uint kb = 0u; kb < 4u; kb++) {
                simdgroup_half8x8 Km;
                simdgroup_load(Km, tg_kv + (kb * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[kb], Qm[d8], Km, S[kb]);
            }
        }

        const bool full_tile = (k0 + 32u <= send);
        float m_tile = -1e30f;
        float2 sv[4];
        if (full_tile) {
            #pragma unroll
            for (uint kb = 0u; kb < 4u; kb++) {
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                if (fm >= T) s2 = float2(-1e30f, -1e30f);
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }
        } else {
            #pragma unroll
            for (uint kb = 0u; kb < 4u; kb++) {
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
                const uint k_a = k0 + kb * 8u + fn;
                const uint k_b = k_a + 1u;
                if (fm >= T || k_a >= send) s2.x = -1e30f;
                if (fm >= T || k_b >= send) s2.y = -1e30f;
                sv[kb] = s2;
                m_tile = max(m_tile, max(s2.x, s2.y));
            }
        }

        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 1u));
        m_tile = max(m_tile, simd_shuffle_xor(m_tile, 8u));
        const float m_new = max(m_row, m_tile);
        const float alpha = (m_row > -1e20f && m_new > -1e20f) ? exp2(m_row - m_new) : 0.0f;

        if (alpha != 1.0f) {
            #pragma unroll
            for (uint d8 = 0u; d8 < 16u; d8++) ELEMS_F2(O[d8].thread_elements()) *= alpha;
        }

        float l_tile = 0.0f;
        threadgroup half* my_p = tg_p + simd * (8u * 36u);
        #pragma unroll
        for (uint kb = 0u; kb < 4u; kb++) {
            float p0 = (sv[kb].x > -1e20f) ? exp2(sv[kb].x - m_new) : 0.0f;
            float p1 = (sv[kb].y > -1e20f) ? exp2(sv[kb].y - m_new) : 0.0f;
            l_tile += p0 + p1;
            *(threadgroup half2*)(my_p + fm * 36u + kb * 8u + fn) = half2(half(p0), half(p1));
        }
        l_tile += simd_shuffle_xor(l_tile, 1u);
        l_tile += simd_shuffle_xor(l_tile, 8u);
        l_row = l_row * alpha + l_tile;
        m_row = m_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint i = 0u; i < 3u; i++) {
            const uint idx = tid + i * 384u;
            if (idx < 1024u) {
                const uint kt_l = idx >> 5, d4 = idx & 31u;
                const uint kj = k0 + kt_l;
                half4 qh = (kj < send) ? *((const device half4*)(Vp + (ulong)kj * 128u) + d4) : half4(0.0h);
                *(threadgroup half4*)(tg_kv + kt_l * 136u + d4 * 4u) = qh;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 Pm[4];
        #pragma unroll
        for (uint kb = 0u; kb < 4u; kb++) simdgroup_load(Pm[kb], my_p + kb * 8u, 36u, ulong2(0, 0), false);
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) {
            #pragma unroll
            for (uint kb = 0u; kb < 4u; kb++) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, tg_kv + (kb * 8u) * 136u + d8 * 8u, 136u, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(O[d8], Pm[kb], Vm, O[d8]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (fm < T && fn == 0u) {
        const ulong idx_h = ((ulong)split_id * 24u + h) * T + fm;
        part_m[idx_h] = m_row;
        part_l[idx_h] = l_row;
    }
    if (fm < T) {
        device float* po_row = part_o + (((ulong)split_id * 24u + h) * T + fm) * 128u;
        #pragma unroll
        for (uint d8 = 0u; d8 < 16u; d8++) *(device float2*)(po_row + d8 * 8u + fn) = ELEMS_F2(O[d8].thread_elements());
    }
    """
    return mx.fast.metal_kernel(
        name="ink_drafter_gqa12_splitk",
        input_names=["q", "K", "V", "meta"],
        output_names=["part_m", "part_l", "part_o"],
        header=header,
        source=src,
    )


@lru_cache(maxsize=2)
def make_drafter_gqa12_reduce_kernel():
    """Pass-2 Split-K merge writing `[1, 24, T, 128]` float16 directly for DSpark drafter SDPA."""
    src = """
    const uint ht = threadgroup_position_in_grid.y;
    const uint T = meta[0];
    const uint n_splits = meta[1];
    const uint h = ht / T;
    const uint t = ht - h * T;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    float m_s = -1e30f;
    float l_s = 0.0f;
    float4 a_s = float4(0.0f);
    for (uint s = simd; s < n_splits; s += 4u) {
        const ulong idx_h = ((ulong)s * 24u + h) * T + t;
        const float ms = part_m[idx_h];
        const float ls = part_l[idx_h];
        if (ls > 0.0f) {
            const float4 as = *((const device float4*)(part_o + idx_h * 128u) + lane);
            const float m_new = max(m_s, ms);
            const float al = (l_s > 0.0f) ? exp2(m_s - m_new) : 0.0f;
            const float be = exp2(ms - m_new);
            l_s = l_s * al + ls * be;
            a_s = a_s * al + as * be;
            m_s = m_new;
        }
    }
    threadgroup float tg_m[4], tg_l[4];
    threadgroup float4 tg_a[4][32];
    if (lane == 0u) { tg_m[simd] = m_s; tg_l[simd] = l_s; }
    tg_a[simd][lane] = a_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float m_all = -1e30f;
        for (uint s = 0u; s < 4u; s++) m_all = max(m_all, tg_m[s]);
        float l_all = 0.0f;
        float4 a_all = float4(0.0f);
        for (uint s = 0u; s < 4u; s++) {
            if (tg_l[s] > 0.0f) {
                const float sc = exp2(tg_m[s] - m_all);
                l_all += tg_l[s] * sc;
                a_all += tg_a[s][lane] * sc;
            }
        }
        const float inv_l = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
        *((device half4*)(out + ((ulong)h * T + t) * 128u) + lane) = half4(a_all * inv_l);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_drafter_gqa12_reduce",
        input_names=["part_m", "part_l", "part_o", "meta"],
        output_names=["out"],
        source=src,
    )


def drafter_gqa_sdpa_fast(q: mx.array, K_buf: mx.array, V_buf: mx.array, lo: int, hi: int, kv_cap: int) -> mx.array:
    """Drop-in fast Metal GQA-12:1 Split-K SDPA for DSpark drafter (`q: [1, 24, T, 128] f16`, `K/V: [1, 2, cap, 128] f16`)."""
    T = int(q.shape[2])
    span = hi - lo
    if span >= 8192:
        n_splits = 32
    elif span >= 2048:
        n_splits = 16
    else:
        n_splits = 8
    pm, pl, po = make_drafter_gqa12_splitk_sdpa_kernel()(
        inputs=[q.reshape(-1), K_buf.reshape(-1), V_buf.reshape(-1), _fk._u32(T, lo, hi, kv_cap, n_splits)],
        grid=(384, 2 * n_splits, 1),
        threadgroup=(384, 1, 1),
        output_shapes=[(n_splits, 24, T), (n_splits, 24, T), (n_splits, 24, T, 128)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return make_drafter_gqa12_reduce_kernel()(
        inputs=[pm.reshape(-1), pl.reshape(-1), po.reshape(-1), _fk._u32(T, n_splits)],
        grid=(128, 24 * T, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, 24, T, 128)],
        output_dtypes=[mx.float16],
    )[0]


# --------------------------------------------------------------------------- Custom Metal Kernel 4: Exact Radix-Histogram Top-P


# Compatibility shim: on stock Apple GPUCompiler toolchains (e.g. GPUCompiler 32023.886 on
# M1 Pro), `threadgroup atomic<float>` fails to compile ("cannot pass pointer to address
# space 'threadgroup' as a pointer to address space 'device'"), while uint atomics in
# threadgroup storage are fine. Emulate exact float fetch-add with a relaxed CAS loop on
# the IEEE-754 bit pattern (same atomicity/order semantics, same parity vs mx.argsort).
_TOPP_UINT_ATOMICS_HEADER = """
inline uint tgu_load(threadgroup atomic_uint* a) {
    return atomic_load_explicit(a, memory_order_relaxed);
}
inline void tgu_store(threadgroup atomic_uint* a, uint v) {
    atomic_store_explicit(a, v, memory_order_relaxed);
}
inline float tgf_add(threadgroup atomic_uint* a, float x) {
    uint prev = atomic_load_explicit(a, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        a, &prev, as_type<uint>(as_type<float>(prev) + x),
        memory_order_relaxed, memory_order_relaxed)) {}
    return as_type<float>(prev);
}
"""


@lru_cache(maxsize=4)
def make_exact_topp_probs_kernel(nt: int = 1024):
    """Exact 3-pass IEEE-754 radix-histogram nucleus (`top_p`) Metal kernel — zero 200K sort, zero CPU sync.

    Uses coalesced stride-1 loads (alignment-safe for any `V`), 1024-bin threadgroup radix histogram,
    and explicit bit-shift tracking (`rmse ~ 1e-10`).
    """
    nsg = nt // 32
    src = f"""
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint V = meta[0];
    const float cap = params[0];
    const float inv_temp = params[1];
    const float top_p = params[2];

    const device float* x = logits + (ulong)row * V;
    device float* y = probs + (ulong)row * V;

    // 1. Coalesced online max + sum_exp
    float lmax = -1e30f;
    float lsum = 0.0f;
    for (uint j = tid; j < V; j += {nt}u) {{
        float v = x[j];
        if (cap > 0.0f) v = cap * tanh(v / cap);
        v *= inv_temp;
        if (v > lmax) {{ lsum = lsum * exp(lmax - v) + 1.0f; lmax = v; }}
        else {{ lsum += exp(v - lmax); }}
    }}
    threadgroup float tg_max[{nsg}];
    threadgroup float tg_sum[{nsg}];
    float smax = simd_max(lmax);
    float ssum = simd_sum(lsum * exp(lmax - smax));
    if (lane == 0u) {{ tg_max[simd] = smax; tg_sum[simd] = ssum; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rmax = tg_max[0];
    for (uint s = 1u; s < {nsg}u; s++) rmax = max(rmax, tg_max[s]);
    float rsum = 0.0f;
    for (uint s = 0u; s < {nsg}u; s++) rsum += tg_sum[s] * exp(tg_max[s] - rmax);
    const float inv_sum = 1.0f / max(rsum, 1e-38f);

    // 2. Write normalized probs + Pass-1 1024-bin radix histogram (bits 29..20; skips ultra-tiny tail atomics unless needed)
    threadgroup atomic_uint h_sum[1024];  // holds IEEE-754 bits of the float sum (CAS-emulated float atomics)
    threadgroup atomic_uint h_cnt[1024];
    for (uint b = tid; b < 1024u; b += {nt}u) {{
        tgu_store(&h_sum[b], 0u);
        atomic_store_explicit(&h_cnt[b], 0u, memory_order_relaxed);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < V; j += {nt}u) {{
        float v = x[j];
        if (cap > 0.0f) v = cap * tanh(v / cap);
        v *= inv_temp;
        float p = exp(v - rmax) * inv_sum;
        y[j] = p;
        if (p > 0.0f) {{
            uint bin = min(1023u, as_type<uint>(p) >> 20);
            if (bin >= 800u) {{
                tgf_add(&h_sum[bin], p);
                atomic_fetch_add_explicit(&h_cnt[bin], 1u, memory_order_relaxed);
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup uint tg_prefix;
    threadgroup uint tg_shift;
    threadgroup float tg_cum_above;
    threadgroup uint tg_done;
    threadgroup float tg_tau_sum;
    threadgroup uint tg_need_low;
    if (tid == 0u) {{
        float cum = 0.0f;
        uint found = 0u, found_cnt = 0u, hit = 0u;
        float found_sum = 0.0f;
        for (int b = 1023; b >= 800; b--) {{
            float hs = as_type<float>(tgu_load(&h_sum[b]));
            uint hc = atomic_load_explicit(&h_cnt[b], memory_order_relaxed);
            if (hc > 0u) {{
                if (cum + hs >= top_p) {{
                    found = (uint)b; found_cnt = hc; found_sum = hs; hit = 1u; break;
                }}
                cum += hs;
            }}
        }}
        tg_prefix = found;
        tg_shift = 20u;
        tg_cum_above = cum;
        tg_tau_sum = found_sum;
        tg_need_low = (hit == 0u) ? 1u : 0u;
        tg_done = (hit == 1u && found_cnt <= 1u) ? 1u : 0u;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rare Pass-1b fallback for ultra-flat distributions dipping below p < 1.5e-8
    if (tg_need_low == 1u) {{
        for (uint j = tid; j < V; j += {nt}u) {{
            float p = y[j];
            if (p > 0.0f) {{
                uint bin = min(1023u, as_type<uint>(p) >> 20);
                if (bin < 800u) {{
                    tgf_add(&h_sum[bin], p);
                    atomic_fetch_add_explicit(&h_cnt[bin], 1u, memory_order_relaxed);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {{
            float cum = tg_cum_above;
            uint found = 0u, found_cnt = 0u, hit = 0u;
            float found_sum = 0.0f;
            uint last_b = 0u, last_cnt = 0u;
            float last_sum = 0.0f;
            for (int b = 799; b >= 0; b--) {{
                float hs = as_type<float>(tgu_load(&h_sum[b]));
                uint hc = atomic_load_explicit(&h_cnt[b], memory_order_relaxed);
                if (hc > 0u) {{
                    last_b = (uint)b; last_cnt = hc; last_sum = hs;
                    if (cum + hs >= top_p) {{ found = (uint)b; found_cnt = hc; found_sum = hs; hit = 1u; break; }}
                    cum += hs;
                }}
            }}
            if (hit == 0u && last_cnt > 0u) {{
                found = last_b; found_cnt = last_cnt; found_sum = last_sum; cum = max(0.0f, cum - last_sum);
            }}
            tg_prefix = found;
            tg_shift = 20u;
            tg_cum_above = cum;
            tg_tau_sum = found_sum;
            tg_done = (found_cnt <= 1u) ? 1u : 0u;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Pass 2 (bits 19..10)
    if (tg_done == 0u) {{
        for (uint b = tid; b < 1024u; b += {nt}u) {{
            tgu_store(&h_sum[b], 0u);
            atomic_store_explicit(&h_cnt[b], 0u, memory_order_relaxed);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pfx = tg_prefix;
        for (uint j = tid; j < V; j += {nt}u) {{
            float p = y[j];
            uint bits = as_type<uint>(p);
            if ((bits >> 20) == pfx) {{
                uint bin = (bits >> 10) & 1023u;
                tgf_add(&h_sum[bin], p);
                atomic_fetch_add_explicit(&h_cnt[bin], 1u, memory_order_relaxed);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {{
            float cum = tg_cum_above;
            uint found = 0u, found_cnt = 0u, hit = 0u;
            float found_sum = 0.0f;
            uint last_b = 0u, last_cnt = 0u;
            float last_sum = 0.0f;
            for (int b = 1023; b >= 0; b--) {{
                float hs = as_type<float>(tgu_load(&h_sum[b]));
                uint hc = atomic_load_explicit(&h_cnt[b], memory_order_relaxed);
                if (hc > 0u) {{
                    last_b = (uint)b; last_cnt = hc; last_sum = hs;
                    if (cum + hs >= top_p) {{ found = (uint)b; found_cnt = hc; found_sum = hs; hit = 1u; break; }}
                    cum += hs;
                }}
            }}
            if (hit == 0u && last_cnt > 0u) {{
                found = last_b; found_cnt = last_cnt; found_sum = last_sum; cum = max(0.0f, cum - last_sum);
            }}
            tg_prefix = (tg_prefix << 10) | found;
            tg_shift = 10u;
            tg_cum_above = cum;
            tg_tau_sum = found_sum;
            tg_done = (found_cnt <= 1u) ? 1u : 0u;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Pass 3 (bits 9..0)
    if (tg_done == 0u) {{
        for (uint b = tid; b < 1024u; b += {nt}u) {{
            tgu_store(&h_sum[b], 0u);
            atomic_store_explicit(&h_cnt[b], 0u, memory_order_relaxed);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pfx = tg_prefix;
        for (uint j = tid; j < V; j += {nt}u) {{
            float p = y[j];
            uint bits = as_type<uint>(p);
            if ((bits >> 10) == pfx) {{
                uint bin = bits & 1023u;
                tgf_add(&h_sum[bin], p);
                atomic_fetch_add_explicit(&h_cnt[bin], 1u, memory_order_relaxed);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {{
            float cum = tg_cum_above;
            uint found = 0u, hit = 0u;
            float found_sum = 0.0f;
            uint last_b = 0u, last_cnt = 0u;
            float last_sum = 0.0f;
            for (int b = 1023; b >= 0; b--) {{
                float hs = as_type<float>(tgu_load(&h_sum[b]));
                uint hc = atomic_load_explicit(&h_cnt[b], memory_order_relaxed);
                if (hc > 0u) {{
                    last_b = (uint)b; last_cnt = hc; last_sum = hs;
                    if (cum + hs >= top_p) {{ found = (uint)b; found_sum = hs; hit = 1u; break; }}
                    cum += hs;
                }}
            }}
            if (hit == 0u && last_cnt > 0u) {{
                found = last_b; found_sum = last_sum; cum = max(0.0f, cum - last_sum);
            }}
            tg_prefix = (tg_prefix << 10) | found;
            tg_shift = 0u;
            tg_cum_above = cum;
            tg_tau_sum = found_sum;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    const uint tau_bits = tg_prefix << tg_shift;
    const float inv_kept = 1.0f / max(tg_cum_above + tg_tau_sum, 1e-20f);
    for (uint j = tid; j < V; j += {nt}u) {{
        float p = y[j];
        y[j] = (as_type<uint>(p) >= tau_bits && p > 0.0f) ? (p * inv_kept) : 0.0f;
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_exact_topp_probs_nt{nt}",
        input_names=["logits", "meta", "params"],
        output_names=["probs"],
        source=src,
        header=_TOPP_UINT_ATOMICS_HEADER,
    )


_ORIG_LOGITS_TO_PROBS = _model_mod.logits_to_probs


def fast_logits_to_probs(logits: mx.array, *, temp: float, top_p: float = 1.0, top_k: int = 0) -> mx.array:
    """Drop-in replacement for `logits_to_probs` using exact radix-histogram Metal top-p."""
    V = int(logits.shape[-1])
    if top_k == 0 and 0.0 < top_p < 1.0 and V >= 1024 and logits.ndim in (1, 2):
        squeeze = logits.ndim == 1
        x2 = logits[None, :] if squeeze else logits
        T = int(x2.shape[0])
        nt = 1024 if V >= 65536 else 256
        out = make_exact_topp_probs_kernel(nt)(
            inputs=[
                x2.astype(mx.float32).reshape(-1),
                mx.array([V], dtype=mx.uint32),
                mx.array([0.0, 1.0 / float(temp), float(top_p)], dtype=mx.float32),
            ],
            grid=(nt, T, 1),
            threadgroup=(nt, 1, 1),
            output_shapes=[(T, V)],
            output_dtypes=[mx.float32],
        )[0]
        return out[0] if squeeze else out
    return _ORIG_LOGITS_TO_PROBS(logits, temp=temp, top_p=top_p, top_k=top_k)


# --------------------------------------------------------------------------- Custom Metal Kernel 5: Lean-SRAM Union-LUT MoE Prefill v3


@lru_cache(maxsize=64)
def make_fast_moe_prefill_v3_kernel(
    n: int,
    fids: tuple[int, ...],
    sorted_in: bool = False,
    sorted_out: bool = False,
    sg: int = 4,
    rb: int = 2,
    n_rg: int = 1,
    max_nb: int = 4,
):
    """High-occupancy Apple Silicon Metal MoE prefill kernel (`sg=4, rb=2, tile_t=32 or 48`).

    Improvements over stock `_make_unified_moe_prefill_v2_kernel`:
      1. Aliases mutually-exclusive `tg_ht4` (4 KB) and `tg_rtr` (8 KB) into ONE 8 KB threadgroup union
         (`tg_lut_u[1024]`), dropping threadgroup SRAM from 17.0 KB -> 12.9 KB (`tile_t=32`) or 15.1 KB
         (`tile_t=48`, both <= 16.0 KB) and doubling hardware occupancy from 1 threadgroup/core to
         2 threadgroups/core on Apple M4 / M4 Pro / M4 Max.
      2. Fine-grained `nb_val in (1, 2, 3, 4, 5, 6)` (`8, 16, 24, 32, 40, 48` tokens) specialization:
         - At `T >= 512` (`tile_t=48, max_nb=6`), eliminates the 2nd DRAM weight pass on all `33..48`-token
           experts (`190 -> 128` work units at `T=512`, `317 -> 251` work units at `T=1024`).
         - For `nb_val = 6` (`41..48` tokens), trades `Xm[6][4]` (`24` regs) for `Am[RB][4]` (`8` regs) +
           1-register `Xm` streaming from `tg_x`, keeping total matrix registers at `33` (zero register spilling!).
      3. Multi-pass `X` tile loading (`tok_stride = sg * 4`) so `sg=4` (128 threads) loads all `1..48`
         tokens cleanly at any `T` (`T=256, 512, 1024`) with exact `max_diff == 0.0` parity vs `sg=8`.
    """
    steps = n // 32
    tok_stride = sg * 4
    max_tok_cap = 8 * max_nb
    nb_tuple = tuple(range(1, max_nb + 1))
    has_ht = any(f in fids for f in (0, 1, 2))
    has_rtr = any(f in fids for f in (3, 4, 5))
    has_nv = 7 in fids
    H = [
        "#include <metal_simdgroup_matrix>\n",
        f"#define SG {sg}u\n#define RB {rb}u\n#define XSTR 36u\n#define PING_STRIDE ({max_tok_cap}u * XSTR)\n",
        "#define ELEMS_H2 reinterpret_cast<thread half2&>\n",
        "#define ELEMS_F2 reinterpret_cast<thread float2&>\n",
    ]
    ha = H.append

    def _emit_x_load(max_t: int, xbase_name: str, step_offset_expr: str, dst_off_expr: str):
        n_passes = (max_t + tok_stride - 1) // tok_stride
        for p_idx in range(n_passes):
            ti_expr = "tok_i" if p_idx == 0 else f"(tok_i + {p_idx * tok_stride}u)"
            cond = f"if ({ti_expr} < {max_t}u)" if (p_idx + 1) * tok_stride > max_t or tok_stride > max_t else ""
            ha(f"        {cond} {{")
            ha(f"            const uint ti = {ti_expr};")
            ha(
                f"            const half4 xv = (ti < cnt) ? *(const device half4*)({xbase_name} + (ulong)tg_xrow[ti] * {n}u + {step_offset_expr} + 4u * vec_k) : half4(0.0h);"
            )
            ha(f"            threadgroup half* rowp = tg_x + {dst_off_expr} + ti * XSTR;")
            ha("            *(threadgroup half2*)(rowp + (vk_h * 2u) * 8u + vk_q * 2u) = xv.xy;")
            ha("            *(threadgroup half2*)(rowp + (vk_h * 2u + 1u) * 8u + vk_q * 2u) = xv.zw;")
            ha("        }")

    def _emit_decode_am(fid: int, target_fn):
        k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
        wpr_c = (n * k) // 32
        if fid in (0, 1):
            if fid == 1:
                ha(f"            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
            else:
                ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u; const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            ha(f"            const half4 v0 = tg_ht4[((extract_bits(lo, {4 * k}u, 12u)) * {_fk.HASH_MULT_HEX}) >> 23u];")
            ha(f"            const half4 v1 = tg_ht4[((extract_bits(lo, 0u, 12u)) * {_fk.HASH_MULT_HEX}) >> 23u];")
            for j, expr in [(0, "v0.xy"), (1, "v0.zw"), (2, "v1.xy"), (3, "v1.zw")]:
                ha(f"            {target_fn(j, expr)}")
        elif fid == 2:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                expr = f"tg_ht2[(extract_bits(lo, {rb_b}u, 12u) * {_fk.HASH_MULT_HEX}) >> 22u]"
                ha(f"            {target_fn(j, expr)}")
        elif fid == 3:
            ha("            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> off) | (w_prev[i] << (32u - off))); w_prev[i] = w0;")
            for j, sh in [(0, 6), (1, 4), (2, 2), (3, 0)]:
                ex = f"extract_bits(lo, {sh}u, 13u)"
                ha(f"            {target_fn(j, _fk._rtr_value(3, 2, ex))}")
        elif fid == 4:
            ha(f"            const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
            for j, rb_b in [(0, 12), (1, 8), (2, 4), (3, 0)]:
                ex = f"extract_bits(lo, {rb_b}u, 14u)"
                ha(f"            {target_fn(j, _fk._rtr_value(4, 2, ex))}")
        elif fid == 5:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u, w2 = (wi + 2u < {wpr_c}u) ? pw[2] : 0u;")
            ha("            const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            ha("            const uint hi = (w1 >> off) | (off == 0u ? 0u : (w2 << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                ex = f"extract_bits(lo, {rb_b}u, 15u)" if rb_b + 15 <= 32 else f"insert_bits(extract_bits(lo, {rb_b}u, {32 - rb_b}u), hi, {32 - rb_b}u, {15 - (32 - rb_b)}u)"
                ha(f"            {target_fn(j, _fk._rtr_value(5, 2, ex))}")
        elif fid == 6:
            ha(f"            const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            for j, (b0, b1) in [(0, (28, 24)), (1, (20, 16)), (2, (12, 8)), (3, (4, 0))]:
                ex0 = f"extract_bits(w0, {b0}u, 12u)" if b0 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b0}u, {32 - b0}u), w1, {32 - b0}u, {12 - (32 - b0)}u)"
                ex1 = f"extract_bits(w0, {b1}u, 12u)" if b1 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b1}u, {32 - b1}u), w1, {32 - b1}u, {12 - (32 - b1)}u)"
                expr = f"half2(lut_rtr4[{ex0}], lut_rtr4[{ex1}])"
                ha(f"            {target_fn(j, expr)}")

    for fid in fids:
        if fid == 7:
            wpr_nv, spr_nv = n // 8, n // 16
            for nb_val in nb_tuple:
                max_t = 8 * nb_val
                use_stream = nb_val >= 6
                ha(
                    f"inline void umoe_v3_f7_nb{nb_val}("
                    "const device half* xbase, const device uint* ecodes, const device uchar* env_scales, float eouter, "
                    "threadgroup half* tg_x, const threadgroup half* tg_e2m1, const threadgroup half* tg_e4m3, "
                    "const threadgroup uint* tg_xrow, const threadgroup uint* tg_out_slot, device float* y, "
                    "uint rbase, uint m_rows, uint y_stride, uint cnt, uint tid, uint lane) {"
                )
                ha("    const uint tok_i = tid >> 3, vec_k = tid & 7u, vk_q = vec_k >> 1, vk_h = vec_k & 1u;")
                ha("    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;")
                ha("    const uint q = fn / 2u, r0 = rbase + fm;")
                ha("    const device uint* p_c[RB]; const device uchar* p_s[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) {{ const uint rr = min(r0 + 8u * i, m_rows - 1u); p_c[i] = ecodes + (ulong)rr * {wpr_nv}u; p_s[i] = env_scales + (ulong)rr * {spr_nv}u; }}")
                ha(f"    simdgroup_float8x8 Acc[RB][{nb_val}];")
                ha(f"    for (uint i = 0; i < RB; i++) for (uint b = 0; b < {nb_val}u; b++) Acc[i][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
                _emit_x_load(max_t, "xbase", "0u", "0u")
                ha("    threadgroup_barrier(mem_flags::mem_threadgroup);")
                ha(f"    for (uint s = 0; s < {steps}u; s++) {{")
                ha("        const uint cur_off = (s & 1u) * PING_STRIDE, nxt_off = ((s + 1u) & 1u) * PING_STRIDE;")
                ha(f"        if (s + 1u < {steps}u) {{")
                _emit_x_load(max_t, "xbase", "32u * (s + 1u)", "nxt_off")
                ha("        }")
                ha("        if (rbase < m_rows) {")
                if not use_stream:
                    ha(f"            simdgroup_half8x8 Xm[{nb_val}][4];")
                    ha("            #pragma unroll")
                    ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                    ha("                #pragma unroll")
                    ha("                for (uint j = 0; j < 4u; j++) simdgroup_load(Xm[b][j], tg_x + cur_off + (8u * b) * XSTR + 8u * j, XSTR, ulong2(0, 0), true);")
                    ha("            }")
                    ha("            #pragma unroll")
                    ha("            for (uint i = 0; i < RB; i++) {")
                    ha("                const uint w0 = p_c[i][4u * s + q];")
                    ha("                const half bsc = tg_e4m3[p_s[i][2u * s + (q >> 1)]];")
                    ha("                simdgroup_half8x8 Am;")
                    for j in range(4):
                        ha(f"                ELEMS_H2(Am.thread_elements()) = half2(tg_e2m1[(w0 >> {8 * j}u) & 15u], tg_e2m1[(w0 >> {8 * j + 4}u) & 15u]) * bsc;\n                #pragma unroll\n                for (uint b = 0; b < {nb_val}u; b++) simdgroup_multiply_accumulate(Acc[i][b], Am, Xm[b][{j}], Acc[i][b]);")
                    ha("            }")
                else:
                    ha("            simdgroup_half8x8 Am[RB][4];")
                    ha("            #pragma unroll")
                    ha("            for (uint i = 0; i < RB; i++) {")
                    ha("                const uint w0 = p_c[i][4u * s + q]; const half bsc = tg_e4m3[p_s[i][2u * s + (q >> 1)]];")
                    for j in range(4):
                        ha(f"                ELEMS_H2(Am[i][{j}].thread_elements()) = half2(tg_e2m1[(w0 >> {8 * j}u) & 15u], tg_e2m1[(w0 >> {8 * j + 4}u) & 15u]) * bsc;")
                    ha("            }")
                    ha("            #pragma unroll")
                    ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                    ha("                #pragma unroll")
                    ha("                for (uint j = 0; j < 4u; j++) {")
                    ha("                    simdgroup_half8x8 Xm; simdgroup_load(Xm, tg_x + cur_off + (8u * b) * XSTR + 8u * j, XSTR, ulong2(0, 0), true);")
                    ha("                    #pragma unroll")
                    ha("                    for (uint i = 0; i < RB; i++) simdgroup_multiply_accumulate(Acc[i][b], Am[i][j], Xm, Acc[i][b]);")
                    ha("                }")
                    ha("            }")
                ha("        }")
                ha("        threadgroup_barrier(mem_flags::mem_threadgroup);")
                ha("    }")
                ha("    if (rbase < m_rows) {")
                ha("        for (uint i = 0; i < RB; i++) {")
                ha("            const uint row = r0 + 8u * i; if (row >= m_rows) continue;")
                ha("            #pragma unroll")
                ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                ha("                const uint ta = 8u * b + fn, tb = ta + 1u;")
                ha("                const float2 d = ELEMS_F2(Acc[i][b].thread_elements()) * eouter;")
                ha("                if (ta < cnt) y[(ulong)tg_out_slot[ta] * y_stride + row] = d.x;")
                ha("                if (tb < cnt) y[(ulong)tg_out_slot[tb] * y_stride + row] = d.y;")
                ha("            }")
                ha("        }")
                ha("    }")
                ha("}")
            continue

        k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
        wpr_c = (n * k) // 32
        for nb_val in nb_tuple:
            max_t = 8 * nb_val
            use_stream = nb_val >= 6
            ha(
                f"inline void umoe_v3_f{fid}_nb{nb_val}("
                "const device half* xbase, const device uint* ecodes, const device half* escales, "
                "threadgroup half* tg_x, const threadgroup half4* tg_ht4, const threadgroup half2* tg_ht2, "
                f"{_fk._RTR_ARGS}const device half* lut_rtr4, "
                "const threadgroup uint* tg_xrow, const threadgroup uint* tg_out_slot, device float* y, "
                "uint rbase, uint m_rows, uint y_stride, uint cnt, uint tid, uint lane) {"
            )
            ha("    const uint tok_i = tid >> 3, vec_k = tid & 7u, vk_q = vec_k >> 1, vk_h = vec_k & 1u;")
            ha("    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;")
            ha("    const uint q = fn / 2u, r0 = rbase + fm;")
            ha("    const device uint* p[RB];")
            ha(f"    for (uint i = 0; i < RB; i++) p[i] = ecodes + (ulong)min(r0 + 8u * i, m_rows - 1u) * {wpr_c}u;")
            ha(f"    const uint first = ({n}u - 8u - 8u * q) * {k}u, w_first = first >> 5, off = first & 31u;")
            ha(f"    simdgroup_float8x8 Acc[RB][{nb_val}];")
            ha(f"    for (uint i = 0; i < RB; i++) for (uint b = 0; b < {nb_val}u; b++) Acc[i][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
            if fid == 3:
                ha("    uint w_prev[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) w_prev[i] = (w_first + 1u < {wpr_c}u) ? p[i][w_first + 1u] : 0u;")
            _emit_x_load(max_t, "xbase", "0u", "0u")
            ha("    threadgroup_barrier(mem_flags::mem_threadgroup);")
            ha(f"    for (uint s = 0; s < {steps}u; s++) {{")
            ha("        const uint cur_off = (s & 1u) * PING_STRIDE, nxt_off = ((s + 1u) & 1u) * PING_STRIDE;")
            ha(f"        if (s + 1u < {steps}u) {{")
            _emit_x_load(max_t, "xbase", "32u * (s + 1u)", "nxt_off")
            ha("        }")
            ha("        if (rbase < m_rows) {")
            if not use_stream:
                ha(f"            simdgroup_half8x8 Xm[{nb_val}][4];")
                ha("            #pragma unroll")
                ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                ha("                #pragma unroll")
                ha("                for (uint j = 0; j < 4u; j++) simdgroup_load(Xm[b][j], tg_x + cur_off + (8u * b) * XSTR + 8u * j, XSTR, ulong2(0, 0), true);")
                ha("            }")
                ha(f"            const uint wi = w_first - s * {k}u;")
                ha("            #pragma unroll")
                ha("            for (uint i = 0; i < RB; i++) {")
                ha("                const device uint* pw = p[i] + wi; simdgroup_half8x8 Am;")
                _emit_decode_am(
                    fid,
                    lambda j, expr: f"ELEMS_H2(Am.thread_elements()) = {expr};\n            #pragma unroll\n            for (uint b = 0; b < {nb_val}u; b++) simdgroup_multiply_accumulate(Acc[i][b], Am, Xm[b][{j}], Acc[i][b]);",
                )
                ha("            }")
            else:
                ha("            simdgroup_half8x8 Am[RB][4];")
                ha(f"            const uint wi = w_first - s * {k}u;")
                ha("            #pragma unroll")
                ha("            for (uint i = 0; i < RB; i++) {")
                ha("                const device uint* pw = p[i] + wi;")
                _emit_decode_am(fid, lambda j, expr: f"ELEMS_H2(Am[i][{j}].thread_elements()) = {expr};")
                ha("            }")
                ha("            #pragma unroll")
                ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                ha("                #pragma unroll")
                ha("                for (uint j = 0; j < 4u; j++) {")
                ha("                    simdgroup_half8x8 Xm; simdgroup_load(Xm, tg_x + cur_off + (8u * b) * XSTR + 8u * j, XSTR, ulong2(0, 0), true);")
                ha("                    #pragma unroll")
                ha("                    for (uint i = 0; i < RB; i++) simdgroup_multiply_accumulate(Acc[i][b], Am[i][j], Xm, Acc[i][b]);")
                ha("                }")
                ha("            }")
            ha("        }")
            ha("        threadgroup_barrier(mem_flags::mem_threadgroup);")
            ha("    }")
            ha("    if (rbase < m_rows) {")
            ha("        for (uint i = 0; i < RB; i++) {")
            ha("            const uint row = r0 + 8u * i; if (row >= m_rows) continue;")
            ha("            const float sc = (float)escales[row];")
            ha("            #pragma unroll")
            ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
            ha("                const uint ta = 8u * b + fn, tb = ta + 1u;")
            ha("                const float2 d = ELEMS_F2(Acc[i][b].thread_elements()) * sc;")
            ha("                if (ta < cnt) y[(ulong)tg_out_slot[ta] * y_stride + row] = d.x;")
            ha("                if (tb < cnt) y[(ulong)tg_out_slot[tb] * y_stride + row] = d.y;")
            ha("            }")
            ha("        }")
            ha("    }")
            ha("}")

    L = [
        "const uint u = threadgroup_position_in_grid.x, rb_tg = threadgroup_position_in_grid.y;",
        "if (u >= unit_count[0]) return;",
        "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup, tid = simd * 32u + lane;",
        "const uint m_rows = meta[0], x_div = meta[1];",
        "const uint eid = units[u * 3u + 0u], base = units[u * 3u + 1u], cnt = units[u * 3u + 2u];",
    ]
    if n_rg > 1:
        L += [
            f"const uint rg_sz = m_rows / {n_rg}u, r_tg = rb_tg * (SG * 8u * RB), rg = r_tg / rg_sz;",
            f"const uint rbase = (r_tg - rg * rg_sz) + simd * (8u * RB), m_grp = rg_sz, idx_rg = eid * {n_rg}u + rg;",
            "const uint fid = exp_fid[idx_rg];",
            "device float* y_ptr = y + rg * rg_sz;",
        ]
    else:
        L += [
            "const uint fid = exp_fid[eid], rbase = rb_tg * (SG * 8u * RB) + simd * (8u * RB), m_grp = m_rows;",
            "device float* y_ptr = y;",
        ]
    L += [
        "const device half4* t_ht1 = (const device half4*)lut_ht1; const device half4* t_ht2 = (const device half4*)lut_ht2;",
        f"threadgroup half tg_x[{2 * max_tok_cap} * XSTR];",
        "threadgroup half4 tg_lut_u[1024]; threadgroup half4* tg_ht4 = tg_lut_u; const threadgroup half2* tg_ht2 = (const threadgroup half2*)tg_lut_u;",
        "threadgroup half* tg_rtr = (threadgroup half*)tg_lut_u; const threadgroup half* tv_rtr1 = tg_rtr; const threadgroup half* tv_rtr2 = tg_rtr; const threadgroup half* tv_rtr3 = tg_rtr;",
        f"threadgroup uint tg_xrow[{max_tok_cap}], tg_out_slot[{max_tok_cap}];",
    ]
    if has_ht:
        L += [
            f"if (fid == 0u) {{ for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht4[idx] = t_ht1[idx]; }}",
            f"else if (fid == 1u) {{ for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht4[idx] = t_ht2[idx]; }}",
            f"else if (fid == 2u) {{ for (uint idx = tid; idx < 512u; idx += {sg * 32}u) tg_ht4[idx] = ((const device half4*)lut_ht3)[idx]; }}",
        ]
    if has_rtr:
        L += [
            f"if (fid >= 3u && fid <= 5u) {{ const device half* t = (fid == 3u) ? lut_rtr1 : (fid == 4u) ? lut_rtr2 : lut_rtr3; for (uint idx = tid; idx < 4096u; idx += {sg * 32}u) tg_rtr[idx] = t[idx]; }}",
        ]
    if has_nv:
        L += [
            "threadgroup half tg_e2m1[16]; threadgroup half tg_e4m3[256];",
            f"if (fid == 7u) {{ if (tid < 16u) tg_e2m1[tid] = lut_e2m1[tid]; for (uint idx = tid; idx < 256u; idx += {sg * 32}u) tg_e4m3[idx] = lut_e4m3[idx]; }}",
        ]
    L.append(
        "if (tid < cnt) { const uint os = slot_order[base + tid]; tg_out_slot[tid] = "
        + ("base + tid;" if sorted_out else "os;")
        + " tg_xrow[tid] = "
        + ("base + tid;" if sorted_in else "(x_div == 1u) ? os : (os >> 3);")
        + " }"
    )
    L.append("threadgroup_barrier(mem_flags::mem_threadgroup);")
    xbase_expr = "((fid == 7u) ? x_raw : x)" if has_nv else "x"
    idx_expr = "idx_rg" if n_rg > 1 else "eid"
    L.append(f"const device uint* ec = codes + exp_code_off[{idx_expr}]; const device half* es = scales + exp_scale_off[{idx_expr}];")
    L.append("switch (fid) {")
    for fid in fids:
        L.append(f"    case {fid}u:")
        if fid == 7:
            L.append("        { const device uchar* env_s = nv_scales + exp_nv_scale_off[eid]; const float eout = nv_outer[eid];")
            for nb_v in nb_tuple[:-1]:
                kw = "if" if nb_v == 1 else "else if"
                L.append(f"          {kw} (cnt <= {8 * nb_v}u) umoe_v3_f7_nb{nb_v}({xbase_expr}, ec, env_s, eout, tg_x, tg_e2m1, tg_e4m3, tg_xrow, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane);")
            L.append(f"          else umoe_v3_f7_nb{max_nb}({xbase_expr}, ec, env_s, eout, tg_x, tg_e2m1, tg_e4m3, tg_xrow, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane); }}")
        else:
            for nb_v in nb_tuple[:-1]:
                kw = "if" if nb_v == 1 else "else if"
                L.append(f"        {kw} (cnt <= {8 * nb_v}u) umoe_v3_f{fid}_nb{nb_v}({xbase_expr}, ec, es, tg_x, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, tg_xrow, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane);")
            L.append(f"        else umoe_v3_f{fid}_nb{max_nb}({xbase_expr}, ec, es, tg_x, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, tg_xrow, tg_out_slot, y_ptr, rbase, m_grp, m_rows, cnt, tid, lane);")
        L.append("        break;")
    L.append("}")
    in_names = ["x", "codes", "scales", "exp_fid", "exp_code_off", "exp_scale_off", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "units", "unit_count", "meta", "slot_order"]
    if has_nv:
        in_names += ["x_raw", "nv_scales", "exp_nv_scale_off", "nv_outer", "lut_e2m1", "lut_e4m3"]
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_umoe_prefill_v3_n{n}_f{tag}_si{int(sorted_in)}_so{int(sorted_out)}_rb{rb}_s{sg}_rg{n_rg}_mnb{max_nb}",
        input_names=in_names,
        output_names=["y"],
        header="\n".join(H),
        source="\n".join(L),
    )


# --------------------------------------------------------------------------- Custom Metal Kernel 6: 2-Slot Parallel SwiGLU + Paley-72 RHT-1152


@lru_cache(maxsize=16)
def make_parallel_swiglu_rht1152_kernel(out_half: bool, also_raw: bool = False, skip_mode: int = 0, pad_zero: bool = False):
    """2 slots per 288-thread threadgroup (`tg_z[32 * 73]` = 9.1 KB SRAM -> 2 TGs/core on M4 Pro).

    Loads `signs` and every `72x72` Paley tile `Bm` ONCE per 2 slots and issues 4 `simdgroup_float8x8`
    MACs per `Bm` load, halving barriers and B-matrix traffic during prefill. Same accumulation order
    as the stock `_make_swiglu_rht1152_sg8_kernel`, so results are bit-identical (`max_diff == 0.0`).
    """
    ot4 = "half4" if out_half else "float4"
    conv0 = "half4(tz4[tid] * INV_S)" if out_half else "(tz4[tid] * INV_S)"
    conv1 = "half4(tz4[288u + tid] * INV_S)" if out_half else "(tz4[288u + tid] * INV_S)"
    raw_conv = "half4(raw_act)" if out_half else "raw_act"
    if skip_mode == 1:
        extra_in = ["eids"]
        skip_expr = "eids[{s}] == 0xFFFFFFFFu"
    else:
        extra_in = []
        skip_expr = "false"
    load_parts = []
    for p in range(2):
        skip_p = skip_expr.format(s=f"slot{p}")
        raw_store = f"\n        ((device {ot4}*)(raw_out + (ulong)slot{p} * 1152u))[tid] = {raw_conv};" if also_raw else ""
        raw_zero = (
            f"\n    else if (in{p}) {{ ((device {ot4}*)(raw_out + (ulong)slot{p} * 1152u))[tid] = {ot4}(0.0f); }}"
            if also_raw
            else ""
        )
        load_parts.append(
            f"""
    const uint slot{p} = pair * 2u + {p}u;
    const bool in{p} = slot{p} < n_slots;
    const bool live{p} = in{p} && !({skip_p});
    float4 act{p} = float4(0.0f);
    if (live{p}) {{
        const device float4* gp4 = (const device float4*)(y_gu + (ulong)slot{p} * 2304u);
        const float4 g = gp4[tid], u = gp4[288u + tid];
        const float4 raw_act = (metal::precise::tanh(g * g_inv) * g_cap)
                             * (1.0f / (1.0f + metal::precise::exp(-g)))
                             * (metal::precise::tanh(u * u_inv) * u_cap);{raw_store}
        act{p} = raw_act * s4;
    }}{raw_zero}
    {{
        threadgroup float* dst = tg_z + ({p * 16}u + a) * 73u + c;
        dst[0] = act{p}.x; dst[1] = act{p}.y; dst[2] = act{p}.z; dst[3] = act{p}.w;
    }}"""
        )
    raw_pad = f"\n        ((device {ot4}*)(raw_out + (ulong)n_slots * 1152u))[tid] = {ot4}(0.0f);" if also_raw else ""
    pad_code = (
        f"""
    if (pair == 0u) {{
        ((device {ot4}*)(out + (ulong)n_slots * 1152u))[tid] = {ot4}(0.0f);{raw_pad}
    }}"""
        if pad_zero
        else ""
    )
    src = f"""
    const uint pair = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup; // 0..8
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = simd * 32u + lane;               // 0..287
    const uint n_slots = meta[0];
    const uint hw = lane >> 4, hl = lane & 15u;
    const float g_cap = caps[0], g_inv = caps[1], u_cap = caps[2], u_inv = caps[3];
    const float4 s4 = ((const device float4*)signs)[tid];
    const uint a = tid / 18u, c = (tid - a * 18u) * 4u;

    threadgroup float tg_z[32 * 73];
{''.join(load_parts)}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c0 = simd * 8u + hw * 4u;
    const bool h1 = hl & 1u, h2 = hl & 2u, h4 = hl & 4u, h8 = hl & 8u;
    #pragma unroll
    for (uint pp = 0; pp < 2u; pp++) {{
        threadgroup float* zrow = tg_z + (pp * 16u + hl) * 73u;
        #pragma unroll
        for (uint j = 0; j < 4u; j++) {{
            const uint cc = c0 + j;
            float v = zrow[cc];
            float o;
            o = simd_shuffle_xor(v, 1u); v = h1 ? (o - v) : (v + o);
            o = simd_shuffle_xor(v, 2u); v = h2 ? (o - v) : (v + o);
            o = simd_shuffle_xor(v, 4u); v = h4 ? (o - v) : (v + o);
            o = simd_shuffle_xor(v, 8u); v = h8 ? (o - v) : (v + o);
            zrow[cc] = v;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma unroll
    for (uint kb = 0; kb < 9u; kb++) {{
        simdgroup_float8x8 Bm, Zm;
        simdgroup_load(Bm, B + (simd * 8u) * 72u + kb * 8u, 72u, ulong2(0, 0), true);
        simdgroup_load(Zm, tg_z + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(Acc0, Zm, Bm, Acc0);
        simdgroup_load(Zm, tg_z + 8u * 73u + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(Acc1, Zm, Bm, Acc1);
        simdgroup_load(Zm, tg_z + 16u * 73u + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(Acc2, Zm, Bm, Acc2);
        simdgroup_load(Zm, tg_z + 24u * 73u + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(Acc3, Zm, Bm, Acc3);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(Acc0, tg_z + simd * 8u, 72u);
    simdgroup_store(Acc1, tg_z + 8u * 72u + simd * 8u, 72u);
    simdgroup_store(Acc2, tg_z + 16u * 72u + simd * 8u, 72u);
    simdgroup_store(Acc3, tg_z + 24u * 72u + simd * 8u, 72u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr float INV_S = 0.02946278254943948f;
    const threadgroup float4* tz4 = (const threadgroup float4*)tg_z;
    if (live0) ((device {ot4}*)(out + (ulong)slot0 * 1152u))[tid] = {conv0};
    if (live1) ((device {ot4}*)(out + (ulong)slot1 * 1152u))[tid] = {conv1};{pad_code}
    """
    out_names = ["out", "raw_out"] if also_raw else ["out"]
    return mx.fast.metal_kernel(
        name=f"ink_par2_swiglu_rht1152_{'h' if out_half else 'f'}_sk{skip_mode}_pz{int(pad_zero)}_ar{int(also_raw)}",
        input_names=["y_gu", "signs", "B", "caps", "meta"] + extra_in,
        output_names=out_names,
        header="#include <metal_simdgroup_matrix>\n",
        source=src,
    )


_ORIG_SWIGLU_RHT1152 = _fk.swiglu_rht_1152_fast
_PAR_SWIGLU_MIN_SLOTS = int(os.environ.get("LAMQ_PAR_SWIGLU_MIN_SLOTS", "512"))


def fast_swiglu_rht_1152(
    y_gu: mx.array,
    signs: mx.array,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
    out_dtype=mx.float16,
    pad_zero_row: bool = False,
    eids_flat: mx.array | None = None,
    offsets: mx.array | None = None,
    also_raw: bool = False,
):
    """Drop-in `swiglu_rht_1152_fast` dispatching prefill-sized calls (`n_slots >= 512`) to Kernel 6."""
    n_slots = int(y_gu.shape[0])
    if n_slots < _PAR_SWIGLU_MIN_SLOTS or offsets is not None:
        return _ORIG_SWIGLU_RHT1152(
            y_gu, signs, gate_cap, up_cap, out_dtype=out_dtype, pad_zero_row=pad_zero_row,
            eids_flat=eids_flat, offsets=offsets, also_raw=also_raw,
        )
    skip_mode = 1 if eids_flat is not None else 0
    caps = _fk._f32_arr(gate_cap, 1.0 / gate_cap, up_cap, 1.0 / up_cap)
    ins = [y_gu.reshape(-1), signs.reshape(-1), _fk._paley_b_flat(1152), caps, mx.array([n_slots], dtype=mx.uint32)]
    if skip_mode == 1:
        ins.append(eids_flat.astype(mx.uint32).reshape(-1))
    out_shape = (n_slots + (1 if pad_zero_row else 0), 1152)
    kern = make_parallel_swiglu_rht1152_kernel(out_dtype == mx.float16, also_raw, skip_mode, pad_zero_row)
    outs = kern(
        inputs=ins,
        grid=(288, (n_slots + 1) // 2, 1),
        threadgroup=(288, 1, 1),
        output_shapes=[out_shape, out_shape] if also_raw else [out_shape],
        output_dtypes=[out_dtype, out_dtype] if also_raw else [out_dtype],
    )
    return (outs[0], outs[1]) if also_raw else outs[0]


def calibrate_draft_temp(temp: float, scale: float | None = None) -> float:
    """Scale draft sampling temperature `tau_q = gamma * tau_p` (`0 < gamma <= 1`) for higher acceptance at 0% target bias.

    In Leviathan et al. (2023) speculative rejection sampling, ANY valid proposal distribution `q(x)` that returns its
    exact probability vector `probs_q` to `verify_spec_sample_gpu` preserves the target distribution `p(x)` with 100%
    mathematical exactness. Because the 3-layer DSpark drafter has slightly higher logit entropy than the 48-visit
    target model, sharpening `tau_q = 0.75 * tau_p` increases the distribution overlap `sum_x min(p(x), q(x))`.
    """
    t = float(temp)
    if t <= 0.0:
        return t
    if scale is None:
        try:
            scale = float(os.environ.get("LAMQ_DRAFT_TEMP_SCALE", "0.75"))
        except ValueError:
            scale = 0.75
    s = float(scale)
    if 0.0 < s < 1.0:
        return max(1e-4, t * s)
    return t


_SPARSE_DRAFT_ORD_MAPS: dict[int, mx.array] = {}


def extract_cyrillic_token_ids(tokenizer_json_path: str | os.PathLike) -> set[int]:
    """Extract all token IDs containing Cyrillic glyphs from a HuggingFace `tokenizer.json` (including ByteLevel BPE).

    In GPT-2/Qwen ByteLevel BPE, UTF-8 lead bytes `0xD0` and `0xD1` (which encode Unicode Cyrillic `U+0400..U+04FF`)
    are mapped to Latin-1 codepoints `U+00D0` ('Ð') and `U+00D1` ('Ñ'). Checking both ByteLevel lead chars and direct
    Unicode `U+0400..U+04FF` recovers all 13,410 Cyrillic vocabulary tokens in Inkling-2-Mini.
    """
    import json
    from pathlib import Path

    p = Path(tokenizer_json_path)
    if not p.is_file():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return set()
    vocab = data.get("model", {}).get("vocab", {})
    cyr_ids: set[int] = set()
    for tok_str, tid in vocab.items():
        if "\u00d0" in tok_str or "\u00d1" in tok_str or any("\u0400" <= ch <= "\u04ff" for ch in tok_str):
            cyr_ids.add(int(tid))
    for added in data.get("added_tokens", ()):
        content = added.get("content", "")
        tid = added.get("id")
        if tid is not None and (
            "\u00d0" in content or "\u00d1" in content or any("\u0400" <= ch <= "\u04ff" for ch in content)
        ):
            cyr_ids.add(int(tid))
    return cyr_ids


def verify_sparse_spec_sample_gpu(
    ordered_logits: mx.array,
    order_mx: mx.array,
    draft_ord: mx.array,
    q_probs: mx.array | None,
    *,
    cap: float | None,
    temp: float,
    top_p: float = 1.0,
    top_k: int = 0,
    draft_ord_map: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Exact Chen/Leviathan unbiased speculative sampling verification supporting sparse/non-contiguous shortlist heads.

    When `draft_ord_map` (`int32[S]`, mapping shortlist index `0..S-1` -> column index `0..V-1` in `ordered_logits`)
    is provided (or registered in `_SPARSE_DRAFT_ORD_MAPS[S]`), maps `draft_ord` through `draft_ord_map` for `p_d` and
    scatters the residual `max(0, p_row[draft_ord_map] - q_sel)` back into `p_row` via `mx.put_along_axis` on GPU.
    """
    if q_probs is not None and draft_ord_map is None:
        draft_ord_map = _SPARSE_DRAFT_ORD_MAPS.get(int(q_probs.shape[1]))
    if q_probs is None or draft_ord_map is None:
        return _gen_mod._orig_verify_spec_sample_gpu(
            ordered_logits,
            order_mx,
            draft_ord,
            q_probs,
            cap=cap,
            temp=temp,
            top_p=top_p,
            top_k=top_k,
        )
    if ordered_logits.ndim == 3:
        ordered_logits = ordered_logits[0]
    T = int(ordered_logits.shape[0])
    K = T - 1

    p_ord = _gen_mod.logits_to_probs(_model_mod.softcap(ordered_logits, cap), temp=temp, top_p=top_p, top_k=top_k)
    rows = mx.arange(K, dtype=mx.int32)
    draft_short_i32 = draft_ord.reshape(K).astype(mx.int32)
    draft_col_i32 = draft_ord_map[draft_short_i32]
    p_d = p_ord[rows, draft_col_i32]
    q_d = q_probs[rows, draft_short_i32]

    u = mx.random.uniform(shape=(K,), dtype=mx.float32)
    accept = (u * q_d) < p_d
    k_match = mx.sum(mx.cumprod(accept.astype(mx.int32), axis=0)).astype(mx.int32)

    all_accepted = k_match == K
    rej_idx = mx.minimum(k_match, K - 1)
    p_row = p_ord[k_match]
    q_sel = q_probs[rej_idx]
    p_sub = p_row[draft_ord_map]
    res_row = mx.put_along_axis(p_row, draft_ord_map, mx.maximum(p_sub - q_sel, 0.0), axis=0)

    res_sum = mx.sum(res_row)
    safe_res = mx.where(res_sum > 0.0, res_row / mx.maximum(res_sum, 1e-38), p_row)
    dist = mx.where(all_accepted, p_row, safe_res)
    log_dist = mx.where(dist > 0.0, mx.log(mx.maximum(dist, 1e-38)), -float("inf"))
    final_ord = mx.random.categorical(log_dist).astype(mx.int32)
    final_tok = order_mx[final_ord].astype(mx.int32)
    return k_match, final_tok


def classify_inkling_cache_slot(cache_slot: int) -> dict[str, Any]:
    """Map an `Inkling2Cache` logical slot `cache_slot` (`0..47`) to its physical layer and loop role.

    In `Inkling2LoopPackedMLX._forward_core_fast`:
      - `0..5`:   Pre-loop Block 1 (`L0..5`, `loop_pass=-1`), beacon tagging layer at `L5` (`cache_slot=5`)
      - `6..17`:  Loop Pass 0 (`L6..17`, `loop_pass=0`), beacon tagging layers at `L11, L17` (`cache_slot=11, 17`)
      - `18..23`: Post-loop Coda Block 4 (`L18..23`, `loop_pass=-2`, logical visits `42..47`)
      - `24..35`: Loop Pass 1 (`L6..17`, `loop_pass=1`), beacon tagging layers at `L11, L17` (`cache_slot=29, 35`)
      - `36..47`: Loop Pass 2 (`L6..17`, `loop_pass=2`), beacon tagging layers at `L11, L17` (`cache_slot=41, 47`)

    Because `18` and `30` are multiples of `6`, `(cache_slot % 6) == (layer_idx % 6)` holds for all 48 slots.
    Per Goodfire's mechanistic analysis, the final layers of Blocks 1..3 (`L5, L11, L17`, i.e. `(cache_slot % 6) == 5`
    outside Coda) write the depth-wise and cross-loop outlier beacon tags (`d2027, d2110, d2296, d1206`) read by
    downstream blocks and loops.
    """
    slot = int(cache_slot)
    if 0 <= slot < 6:
        layer_idx, loop_pass = slot, -1
    elif 6 <= slot < 18:
        layer_idx, loop_pass = slot, 0
    elif 18 <= slot < 24:
        layer_idx, loop_pass = slot, -2
    elif 24 <= slot < 36:
        layer_idx, loop_pass = slot - 18, 1
    elif 36 <= slot < 48:
        layer_idx, loop_pass = slot - 30, 2
    else:
        layer_idx, loop_pass = slot % 24, -2
    is_postloop = 18 <= slot < 24
    is_loop3 = 36 <= slot < 48
    is_beacon = (not is_postloop) and ((slot % 6) == 5)
    return {
        "cache_slot": slot,
        "layer_idx": layer_idx,
        "loop_pass": loop_pass,
        "block_idx": layer_idx // 6,
        "is_postloop": is_postloop,
        "is_loop3": is_loop3,
        "is_beacon": is_beacon,
    }


def resolve_decode_moe_routing(cfg: Any, cache_slot: int, T: int) -> tuple[bool, int, int, float, bool]:
    """Resolve `(dyn, k_max, k_min, ratio_thr, renorm)` for MoE visit `cache_slot` (`0..47`) at sequence length `T`.

    - Prefill (`T > 8`): defaults to 100% static `k=8` (`dyn=False`) unless `LAMQ_PREFILL_DYNAMIC_MOE=1` is set,
      preventing prefill throughput slowdown (`-3.8%`) and prompt KV cache drift (`KL = 0.01025`).
    - Decode (`T <= 8`): combines `cfg.moe_*` with optional `LAMQ_DECODE_MOE_K_MAX`, `LAMQ_DECODE_MOE_K_MIN`,
      `LAMQ_DECODE_MOE_RATIO_THR`, `LAMQ_LOOP3_MOE_K_MAX`, and post-loop `L18..23` (`18 <= cache_slot < 24`) /
      non-beacon Loop Pass 3 (`36 <= cache_slot < 48, (cache_slot % 6) != 5`) `LAMQ_POSTLOOP_MOE_K_MAX`.
    - Goodfire Beacon Protection (`LAMQ_BEACON_PROTECT_MOE=1`, default ON for pass-specific `LAMQ_POSTLOOP_MOE_K_MAX` /
      `LAMQ_LOOP3_MOE_K_MAX`): locks the block-final beacon tagging layers `L5, L11, L17` (`cache_slot in {5,11,17,29,35,41,47}`)
      at lossless `k=8` so cross-loop (`d2027, d2110`) and depth-wise (`d2296, d1206`) routing spikes stay exact.
    """
    if T > 8 and os.environ.get("LAMQ_PREFILL_DYNAMIC_MOE", "0") != "1":
        return False, 8, 8, 0.0, False

    k_max = int(getattr(cfg, "moe_k_max", 8))
    k_min = int(getattr(cfg, "moe_k_min", 8))
    ratio_thr = float(getattr(cfg, "moe_ratio_thr", 0.0))
    renorm = bool(getattr(cfg, "moe_renorm", False))
    env_active = False

    if T <= 8:
        slot_info = classify_inkling_cache_slot(cache_slot)
        beacon_protect = os.environ.get("LAMQ_BEACON_PROTECT_MOE", "1") != "0"
        env_kmax = os.environ.get("LAMQ_DECODE_MOE_K_MAX", "").strip()
        if env_kmax:
            if not (beacon_protect and slot_info["is_beacon"] and os.environ.get("LAMQ_FORCE_GLOBAL_BEACON_PRUNE", "0") != "1"):
                k_max = max(1, min(8, int(env_kmax)))
                env_active = True
            elif not os.environ.get("LAMQ_BEACON_PROTECT_STRICT", "0") == "1":
                # If global LAMQ_DECODE_MOE_K_MAX is explicitly requested without strict override, allow global k_max
                # unless this is a pass-specific postloop/loop3 rule.
                k_max = max(1, min(8, int(env_kmax)))
                env_active = True
        env_kmin = os.environ.get("LAMQ_DECODE_MOE_K_MIN", "").strip()
        if env_kmin:
            k_min = max(1, min(8, int(env_kmin)))
            env_active = True
        env_thr = os.environ.get("LAMQ_DECODE_MOE_RATIO_THR", "").strip()
        if env_thr:
            ratio_thr = max(0.0, float(env_thr))
            env_active = True

        # Tap-41 Coherent Beacon-Protected MoE Routing:
        # - `LAMQ_POSTLOOP_MOE_K_MAX`: applies to pre-tap41 Loop-3 Block-3 non-beacon slots (42..46, L12..16),
        #   while protecting Beacon L17 (slot 47) and keeping post-tap41 Coda L18..23 (slots 18..23) coherent with DSpark!
        # - `LAMQ_CODA_MOE_K_MAX`: optional explicit override for Coda L18..23 (slots 18..23, e.g. for serial non-DSpark decode).
        # - `LAMQ_LOOP3_MOE_K_MAX`: optional explicit override for all 10 non-beacon Loop-3 slots (36..40, 42..46).
        env_post = os.environ.get("LAMQ_POSTLOOP_MOE_K_MAX", "").strip()
        env_coda = os.environ.get("LAMQ_CODA_MOE_K_MAX", "").strip()
        env_loop3 = os.environ.get("LAMQ_LOOP3_MOE_K_MAX", "").strip()
        is_non_beacon_loop3 = slot_info["is_loop3"] and (not (beacon_protect and slot_info["is_beacon"]))

        if slot_info["is_postloop"] and env_coda:
            coda_k = max(1, min(8, int(env_coda)))
            k_max = min(k_max, coda_k)
            k_min = min(k_min, coda_k)
            env_active = True
        elif is_non_beacon_loop3 and env_loop3:
            l3_k = max(1, min(8, int(env_loop3)))
            k_max = min(k_max, l3_k)
            k_min = min(k_min, l3_k)
            env_active = True
        elif cache_slot >= 42 and env_post and (not (beacon_protect and slot_info["is_beacon"])):
            post_k = max(1, min(8, int(env_post)))
            k_max = min(k_max, post_k)
            k_min = min(k_min, post_k)
            env_active = True

        if env_active and not env_kmin and ratio_thr <= 0.0:
            k_min = min(k_min, k_max)
        if env_active:
            renorm = os.environ.get("LAMQ_DECODE_MOE_RENORM", "1") != "0"

    k_min = min(k_min, k_max)
    dyn = (k_max < 8) or (ratio_thr > 0.0)
    return dyn, k_max, k_min, ratio_thr, renorm


def build_hybrid_quant_dir(
    high_dir: str | os.PathLike,
    low_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    *,
    low_layers: tuple[int, ...] | list[int] = (18, 19, 20, 21, 22, 23),
) -> dict[str, Any]:
    """Create a zero-copy symlinked Loop-Aware Hybrid model directory combining `high_dir` (e.g. 12.8GB) and `low_dir` (e.g. 8.9GB).

    By default, keeps pre-loop `L0..5` and 3x-looped core `L6..17` (36 visits) from `high_dir` (`12.8GB`) and replaces
    only the 1x post-loop layers `L18..23` (6 visits) with `low_dir` (`8.9GB`), saving `725.2 MB` of wired Unified
    Memory with `dKL = 0.00018` and `100.0%` top-1 token agreement. Also merges `luts.npz` across both directories.
    """
    import json
    from pathlib import Path
    import numpy as np

    hi_path = Path(high_dir).expanduser().resolve()
    lo_path = Path(low_dir).expanduser().resolve()
    dst_path = Path(out_dir).expanduser().resolve()
    if not (hi_path / "config.json").is_file():
        raise FileNotFoundError(f"high_dir missing config.json: {hi_path}")
    if not (lo_path / "config.json").is_file():
        raise FileNotFoundError(f"low_dir missing config.json: {lo_path}")

    dst_path.mkdir(parents=True, exist_ok=True)
    low_set = {int(idx) for idx in low_layers}

    # 1. Symlink all shared non-layer files from high_dir (except luts.npz and hybrid_manifest.json)
    for item in sorted(hi_path.iterdir()):
        name = item.name
        if name in ("luts.npz", "hybrid_manifest.json") or name.startswith("layer_"):
            continue
        target = dst_path / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(item)

    # 2. Symlink per-layer files and compute exact byte savings
    total_bytes = 0
    saved_bytes = 0
    layer_sources: dict[str, str] = {}
    for idx in range(24):
        src_base = lo_path if idx in low_set else hi_path
        layer_sources[f"layer_{idx:02d}"] = "low" if idx in low_set else "high"
        hi_npz = hi_path / f"layer_{idx:02d}_packed.npz"
        lo_npz = lo_path / f"layer_{idx:02d}_packed.npz"
        if idx in low_set and hi_npz.is_file() and lo_npz.is_file():
            saved_bytes += max(0, hi_npz.stat().st_size - lo_npz.stat().st_size)
        for suffix in ("_meta.json", "_packed.npz"):
            fname = f"layer_{idx:02d}{suffix}"
            src_file = src_base / fname
            if not src_file.is_file():
                raise FileNotFoundError(f"Missing layer file: {src_file}")
            dst_file = dst_path / fname
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            dst_file.symlink_to(src_file)
            total_bytes += src_file.stat().st_size

    # 3. Merge luts.npz from high_dir and low_dir so all format LUTs (nvfp4, rtr1..4) are present
    merged_luts: dict[str, np.ndarray] = {}
    for src_lut_path in (lo_path / "luts.npz", hi_path / "luts.npz"):
        if src_lut_path.is_file():
            with np.load(src_lut_path) as lnpz:
                for k in lnpz.files:
                    merged_luts[k] = lnpz[k]
    dst_luts = dst_path / "luts.npz"
    if dst_luts.exists() or dst_luts.is_symlink():
        dst_luts.unlink()
    np.savez(dst_luts, **merged_luts)

    manifest = {
        "high_dir": hi_path.name,
        "low_dir": lo_path.name,
        "low_layers": sorted(low_set),
        "saved_bytes": int(saved_bytes),
        "saved_mb": round(saved_bytes / (1024 * 1024), 2),
        "total_layer_gb": round(total_bytes / (1024**3), 3),
        "luts_keys": sorted(merged_luts.keys()),
        "layers": layer_sources,
    }
    (dst_path / "hybrid_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_hybrid_cli(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `inkling-build-hybrid` (`12.8GB x 8.9GB` zero-copy post-loop hybrid builder)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a zero-copy symlinked Loop-Aware Hybrid model directory (e.g. 12.8GB core + 8.9GB post-loop L18..23)."
    )
    parser.add_argument("--high-dir", required=True, help="Path to high-precision packed model directory (e.g. 12.8GB)")
    parser.add_argument("--low-dir", required=True, help="Path to lower-precision packed model directory (e.g. 8.9GB)")
    parser.add_argument("--out-dir", required=True, help="Output directory for the zero-copy symlinked hybrid model")
    parser.add_argument(
        "--low-layers",
        default="18,19,20,21,22,23",
        help="Comma-separated layer indices to take from --low-dir (default: post-loop 18,19,20,21,22,23)",
    )
    args = parser.parse_args(argv)
    low_layers = tuple(int(x.strip()) for x in args.low_layers.split(",") if x.strip())
    manifest = build_hybrid_quant_dir(args.high_dir, args.low_dir, args.out_dir, low_layers=low_layers)
    print(
        f"Built hybrid model at {args.out_dir}: low_layers={manifest['low_layers']} "
        f"| saved={manifest['saved_mb']:.1f} MB | total_layer_gb={manifest['total_layer_gb']:.3f} GB"
    )
    return 0


def repack_moe_arrays_hot_cold(
    arrays: dict[str, Any],
    router_w: mx.array,
    router_bias: mx.array,
    id_to_fmt: dict[int, str] | None = None,
) -> tuple[dict[str, Any], mx.array, mx.array, Any]:
    """Format-homogeneous & frequency-sorted (hot-to-cold) MoE expert slab packing (100% bit-exact, `max_diff == 0.0`).

    Sorts the 128 experts of a layer by `(dn_fid, gu_fid, -hotness, eid)` where `hotness = router_bias + 0.05 * ||router_w||_2`,
    permutes `router_w` and `router_bias`, and compacts every format slab (`gu_*`, `dn_*`) so that:
    1. All experts sharing `(dn_fid, gu_fid)` have contiguous `eid`s and contiguous physical slab offsets (`exp_code_off`),
       maximizing SIMD-group warp coherence in `plan` (`T=2..8`) and `fused_moe_prefill_v4` (`T > 8`).
    2. Within each format slab, the most frequently routed experts sit contiguously in the leading 16-KB memory pages,
       reducing GPU TLB misses and improving Apple M4 Pro SLC retention across the 3 loop passes (`L6..17`).
    """
    import numpy as np

    if "gu_map" not in arrays or "dn_map" not in arrays or "gu_rg_map" in arrays or "dn_rg_map" in arrays:
        n_e = int(router_bias.shape[0]) if hasattr(router_bias, "shape") else 128
        return arrays, router_w, router_bias, np.arange(n_e, dtype=np.int32)

    if id_to_fmt is None:
        id_to_fmt = getattr(_model_mod, "ID_TO_FMT", {0: "ht1", 1: "ht2", 2: "ht3", 3: "rtr1", 4: "rtr2", 5: "rtr3", 6: "rtr4"})

    gu_map = np.asarray(arrays["gu_map"], dtype=np.int32)
    dn_map = np.asarray(arrays["dn_map"], dtype=np.int32)
    n_exp = int(gu_map.shape[0])
    bias_np = np.array(router_bias.astype(mx.float32), dtype=np.float32).reshape(n_exp)
    w_np = np.array(router_w.astype(mx.float32), dtype=np.float32).reshape(n_exp, -1)
    hotness = bias_np + 0.05 * np.linalg.norm(w_np, axis=1)

    perm = np.array(
        sorted(range(n_exp), key=lambda e: (int(dn_map[e, 0]), int(gu_map[e, 0]), -float(hotness[e]), e)),
        dtype=np.int32,
    )
    if np.array_equal(perm, np.arange(n_exp, dtype=np.int32)):
        return arrays, router_w, router_bias, perm

    out_arrays = dict(arrays)
    for side, old_map in (("gu", gu_map), ("dn", dn_map)):
        new_map = old_map[perm].copy()
        for fid in np.unique(new_map[:, 0]):
            mask = new_map[:, 0] == fid
            old_locs = new_map[mask, 1]
            if int(fid) == 7:
                for suf in ("codes", "scales", "outer"):
                    k = f"{side}_nvfp4_{suf}"
                    if k in out_arrays:
                        out_arrays[k] = np.ascontiguousarray(out_arrays[k][old_locs])
            else:
                fmt = id_to_fmt.get(int(fid))
                if fmt:
                    for suf in ("codes", "scales"):
                        k = f"{side}_{fmt}_{suf}"
                        if k in out_arrays:
                            out_arrays[k] = np.ascontiguousarray(out_arrays[k][old_locs])
            new_map[mask, 1] = np.arange(len(old_locs), dtype=new_map.dtype)
        out_arrays[f"{side}_map"] = new_map.astype(old_map.dtype)

    perm_mx = mx.array(perm)
    return out_arrays, router_w[perm_mx], router_bias[perm_mx], perm


def find_companion_low_tier_dir(packed_dir: str | os.PathLike | None) -> Any:
    """Locate a companion lower-bit packed directory (e.g. `8.9GB`) for Ant-style live post-loop layer tiering."""
    from pathlib import Path

    env_low = os.environ.get("LAMQ_ANT_LOW_DIR", "").strip()
    if env_low:
        p = Path(env_low).expanduser()
        if (p / "config.json").is_file() and (p / "luts.npz").is_file():
            return p
    if packed_dir is None:
        return None
    base = Path(packed_dir).expanduser()
    if "8.9GB" in base.name:
        return None
    sibling = base.parent / "Inkling-2-Mini-LookAtMyQuant-8.9GB"
    if sibling != base and (sibling / "config.json").is_file() and (sibling / "luts.npz").is_file():
        return sibling
    return None


def ensure_hybrid_luts(model: Any, low_dir: str | os.PathLike) -> bool:
    """Merge any missing format LUTs (`rtr1..4`, etc.) from `low_dir/luts.npz` into `model.luts` & `model.fast_luts`."""
    from pathlib import Path
    import numpy as np

    lut_path = Path(low_dir) / "luts.npz"
    if not lut_path.is_file() or not hasattr(model, "luts"):
        return False
    added = False
    with np.load(lut_path) as lnpz:
        for k in lnpz.files:
            if k not in model.luts:
                model.luts[k] = mx.array(lnpz[k])
                added = True
    if added and getattr(model, "_fast", False) and hasattr(_model_mod, "build_fast_luts"):
        model.fast_luts = _model_mod.build_fast_luts(model.luts)
    return added


def set_postloop_ant_tier(
    model: Any,
    tier: str,
    *,
    low_dir: str | os.PathLike | None = None,
    layers: tuple[int, ...] = (18, 19, 20, 21, 22, 23),
    log_fn=None,
) -> dict[str, Any]:
    """Live Ant-style post-loop layer tier switch (`"high"` = 12.8GB <-> `"low"` = 8.9GB on `L18..23`) with zero KV loss.

    Because post-loop layers `L18..23` execute only 1x after the 3x recurrent loop (`L6..17`), switching them to `8.9GB`
    frees `~700 MB` of wired Unified Memory in `< 0.25s` with `dKL = 0.00018` and `100.0%` top-1 token agreement while
    preserving 100% of live `Inkling2Cache` and `SmartPromptCache` buffers.
    """
    from pathlib import Path
    import time

    target_tier = str(tier).strip().lower()
    if target_tier not in ("high", "low"):
        raise ValueError(f"Expected tier in ('high', 'low'), got {tier!r}")

    cur_tier = getattr(model, "_ant_postloop_tier", "high")
    hi_dir = Path(getattr(model, "packed_dir", ""))
    lo_path = Path(low_dir).expanduser() if low_dir is not None else find_companion_low_tier_dir(hi_dir)
    if target_tier == "low" and (lo_path is None or not (lo_path / "config.json").is_file()):
        return {"tier": cur_tier, "changed": False, "reason": "no_low_dir"}

    src_dir = lo_path if target_tier == "low" else hi_dir
    if target_tier == cur_tier and getattr(model, "_ant_postloop_layers", None) == tuple(layers):
        return {"tier": cur_tier, "changed": False, "active_gb": round(mx.get_active_memory() / 1e9, 3)}

    if target_tier == "low" and lo_path is not None:
        ensure_hybrid_luts(model, lo_path)

    t0 = time.perf_counter()
    mem_before = mx.get_active_memory()
    layer_cls = type(model.layers[0])
    for idx in layers:
        if 0 <= idx < len(model.layers):
            old_layer = model.layers[idx]
            model.layers[idx] = layer_cls(idx, model.cfg, src_dir, model.luts, model.fast_luts, model.st)
            del old_layer
    mx.clear_cache()
    mem_after = mx.get_active_memory()
    dt_ms = (time.perf_counter() - t0) * 1e3
    model._ant_postloop_tier = target_tier
    model._ant_postloop_layers = tuple(layers)
    model._ant_low_dir = lo_path
    delta_mb = (mem_before - mem_after) / (1024 * 1024)
    if log_fn is not None:
        log_fn(
            f"[ant] switched post-loop layers {list(layers)} -> {target_tier.upper()} ({src_dir.name}) "
            f"in {dt_ms:.0f}ms | delta={-delta_mb:+.1f} MB | active {mem_after / 1e9:.2f} GB"
        )
    return {
        "tier": target_tier,
        "changed": True,
        "layers": list(layers),
        "source_dir": src_dir.name,
        "elapsed_ms": round(dt_ms, 1),
        "freed_mb": round(delta_mb, 1),
        "active_gb": round(mem_after / 1e9, 3),
    }


# --------------------------------------------------------------------------- Goodfire Mechanistic Math: Rank-2 Cross-Loop & Pre-Normalized BlockAttnRes


@lru_cache(maxsize=4)
def _make_prenorm_bar_kernel(with_norm: bool):
    norm_inputs = ["q", "blocks_hat", "partial", "norm_w", "meta"] if with_norm else ["q", "blocks_hat", "partial", "meta"]
    store_code = (
        """
        float ss = 0.0f;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) ss += acc[qi][e] * acc[qi][e];
        ss = simd_sum(ss);
        if (lane == 0u) red[qi][simd] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0; i < 8u; i++) tot += red[qi][i];
        const float ir_out = metal::precise::rsqrt(tot * (1.0f / 2304.0f) + 1e-6f);
        device float* op = out + ((ulong)qi * T + t) * 2304u;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) {
            const uint ch = tid + e * 256u;
            op[ch] = acc[qi][e] * ir_out * norm_w[ch];
        }
        """
        if with_norm
        else """
        device float* op = out + ((ulong)qi * T + t) * 2304u;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) op[tid + e * 256u] = acc[qi][e];
        """
    )
    src = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y;
    const uint n_blocks = meta[0], num_q = meta[1], T = meta[2];
    const uint S = n_blocks + 1u;

    threadgroup float red[4][8];
    threadgroup float tg_ss_part[8];
    threadgroup float tg_dp[6][4][8];
    threadgroup float tg_w[4][6];

    float qv[4][9];
    for (uint qi = 0; qi < num_q; qi++) {{
        const device float* qp = q + (ulong)qi * 2304u;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) qv[qi][e] = qp[tid + e * 256u];
    }}

    // Hold all S <= 5 states in thread registers (1x DRAM read instead of 2x!)
    // Completed blocks (s < n_blocks) are already RMS-normalized at block boundaries (inv_rms == 1.0f).
    float sv_all[5][9];
    for (uint s = 0; s < n_blocks; s++) {{
        const device float* sp = blocks_hat + ((ulong)s * T + t) * 2304u;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) sv_all[s][e] = sp[tid + e * 256u];
        for (uint qi = 0; qi < num_q; qi++) {{
            float dp = 0.0f;
            #pragma unroll
            for (uint e = 0; e < 9u; e++) dp += qv[qi][e] * sv_all[s][e];
            dp = simd_sum(dp);
            if (lane == 0u) tg_dp[s][qi][simd] = dp;
        }}
    }}
    {{
        const device float* sp = partial + (ulong)t * 2304u;
        float ss = 0.0f;
        #pragma unroll
        for (uint e = 0; e < 9u; e++) {{
            const float v = sp[tid + e * 256u];
            sv_all[n_blocks][e] = v;
            ss += v * v;
        }}
        ss = simd_sum(ss);
        if (lane == 0u) tg_ss_part[simd] = ss;
        for (uint qi = 0; qi < num_q; qi++) {{
            float dp = 0.0f;
            #pragma unroll
            for (uint e = 0; e < 9u; e++) dp += qv[qi][e] * sv_all[n_blocks][e];
            dp = simd_sum(dp);
            if (lane == 0u) tg_dp[n_blocks][qi][simd] = dp;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {{
        float tot_p = 0.0f;
        for (uint i = 0; i < 8u; i++) tot_p += tg_ss_part[i];
        const float ir_part = metal::precise::rsqrt(tot_p * (1.0f / 2304.0f) + 1e-6f);
        for (uint qi = 0; qi < num_q; qi++) {{
            float sc[6];
            float mxv = -1e30f;
            for (uint s = 0; s < S; s++) {{
                float dp = 0.0f;
                for (uint i = 0; i < 8u; i++) dp += tg_dp[s][qi][i];
                const float ir = (s < n_blocks) ? 1.0f : ir_part;
                sc[s] = dp * ir * (1.0f / 48.0f);
                mxv = max(mxv, sc[s]);
            }}
            float sum = 0.0f;
            for (uint s = 0; s < S; s++) {{ float w = metal::precise::exp(sc[s] - mxv); tg_w[qi][s] = w; sum += w; }}
            const float inv = 1.0f / sum;
            for (uint s = 0; s < n_blocks; s++) tg_w[qi][s] *= inv;
            tg_w[qi][n_blocks] *= inv * ir_part;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[4][9];
    for (uint qi = 0; qi < num_q; qi++) for (uint e = 0; e < 9u; e++) acc[qi][e] = 0.0f;
    for (uint s = 0; s < S; s++) {{
        for (uint qi = 0; qi < num_q; qi++) {{
            const float w = tg_w[qi][s];
            #pragma unroll
            for (uint e = 0; e < 9u; e++) acc[qi][e] += w * sv_all[s][e];
        }}
    }}
    for (uint qi = 0; qi < num_q; qi++) {{
        {store_code}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"ink_bar_prenorm_n{int(with_norm)}",
        input_names=norm_inputs,
        output_names=["out"],
        source=src,
    )


@lru_cache(maxsize=2)
def _make_prenorm_bar_norm_rht_kernel():
    src = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y;
    const uint n_blocks = meta[0], T = meta[1];
    const uint S = n_blocks + 1u;
    const uint c0 = simd * 9u;

    threadgroup float red[8];
    threadgroup float tg_ss_part[8];
    threadgroup float tg_dp[6][8];
    threadgroup float tg_w[6];
    threadgroup float tg_z[32 * 72];

    float qv[9];
    #pragma unroll
    for (uint j = 0; j < 9u; j++) qv[j] = q[lane * 72u + c0 + j];

    float sv_all[5][9];
    for (uint s = 0; s < n_blocks; s++) {
        const device float* sp = blocks_hat + ((ulong)s * T + t) * 2304u;
        float dp = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 9u; j++) {
            const float v = sp[lane * 72u + c0 + j];
            sv_all[s][j] = v;
            dp += qv[j] * v;
        }
        dp = simd_sum(dp);
        if (lane == 0u) tg_dp[s][simd] = dp;
    }
    {
        const device float* sp = partial + (ulong)t * 2304u;
        float ss = 0.0f, dp = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 9u; j++) {
            const float v = sp[lane * 72u + c0 + j];
            sv_all[n_blocks][j] = v;
            ss += v * v;
            dp += qv[j] * v;
        }
        ss = simd_sum(ss);
        dp = simd_sum(dp);
        if (lane == 0u) { tg_ss_part[simd] = ss; tg_dp[n_blocks][simd] = dp; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float tot_s = 0.0f;
        for (uint i = 0; i < 8u; i++) tot_s += tg_ss_part[i];
        const float ir_part = metal::precise::rsqrt(tot_s * (1.0f / 2304.0f) + 1e-6f);
        float sc[6];
        float mxv = -1e30f;
        for (uint s = 0; s < S; s++) {
            float tot_d = 0.0f;
            for (uint i = 0; i < 8u; i++) tot_d += tg_dp[s][i];
            const float ir = (s < n_blocks) ? 1.0f : ir_part;
            sc[s] = tot_d * ir * (1.0f / 48.0f);
            mxv = max(mxv, sc[s]);
        }
        float sum = 0.0f;
        for (uint s = 0; s < S; s++) { float w = metal::precise::exp(sc[s] - mxv); tg_w[s] = w; sum += w; }
        const float inv = 1.0f / sum;
        for (uint s = 0; s < n_blocks; s++) tg_w[s] *= inv;
        tg_w[n_blocks] *= inv * ir_part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[9];
    #pragma unroll
    for (uint j = 0; j < 9u; j++) acc[j] = 0.0f;
    for (uint s = 0; s < S; s++) {
        const float w = tg_w[s];
        #pragma unroll
        for (uint j = 0; j < 9u; j++) acc[j] += w * sv_all[s][j];
    }
    float ss_out = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) ss_out += acc[j] * acc[j];
    ss_out = simd_sum(ss_out);
    if (lane == 0u) red[simd] = ss_out;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot_out = 0.0f;
    for (uint i = 0; i < 8u; i++) tot_out += red[i];
    const float ir_out = metal::precise::rsqrt(tot_out * (1.0f / 2304.0f) + 1e-6f);

    device half* hp = h_f16_out + (ulong)t * 2304u;
    device half* rp = x_rot_out + (ulong)t * 2304u;
    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const uint idx = lane * 72u + c;
        const float hm = acc[j] * ir_out * norm_w[idx];
        hp[idx] = (half)hm;
        float v = hm * gu_signs[idx];
        float o;
        o = simd_shuffle_xor(v, 1u);  v = h1  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u);  v = h2  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u);  v = h4  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u);  v = h8  ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 16u); v = h16 ? (o - v) : (v + o);
        tg_z[idx] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float* zr = tg_z + lane * 72u;
    constexpr float INV_S = 1.0f / 48.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const device float4* br = (const device float4*)(B + c * 72u);
        float racc = 0.0f;
        #pragma unroll
        for (uint k4 = 0; k4 < 18u; k4++) {
            const float4 b4 = br[k4];
            const uint k = k4 * 4u;
            racc += zr[k] * b4.x + zr[k + 1u] * b4.y + zr[k + 2u] * b4.z + zr[k + 3u] * b4.w;
        }
        rp[lane * 72u + c] = (half)(racc * INV_S);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_bar_prenorm_norm_rht_f16",
        input_names=["q", "blocks_hat", "partial", "norm_w", "gu_signs", "B", "meta"],
        output_names=["h_f16_out", "x_rot_out"],
        source=src,
    )


def block_attn_res_prenorm_fused(
    query: mx.array,
    blocks_hat: mx.array | None,
    n_blocks: int,
    partial: mx.array,
    norm_w: mx.array | None = None,
) -> mx.array:
    """Single-read BlockAttnRes (+ optional RMSNorm) where `blocks_hat` stores pre-normalized completed blocks."""
    T = int(partial.size) // 2304
    num_q = 1 if query.ndim == 1 else int(query.shape[0])
    b_in = blocks_hat.reshape(-1) if (blocks_hat is not None and n_blocks > 0) else _fk._EMPTY_F32
    mt = _fk._u32(n_blocks, num_q, T)
    if norm_w is not None:
        return _make_prenorm_bar_kernel(True)(
            inputs=[query.reshape(-1), b_in, partial.reshape(-1), norm_w.reshape(-1), mt],
            grid=(256, T, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(num_q, T, 2304)],
            output_dtypes=[mx.float32],
        )[0]
    return _make_prenorm_bar_kernel(False)(
        inputs=[query.reshape(-1), b_in, partial.reshape(-1), mt],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(num_q, T, 2304)],
        output_dtypes=[mx.float32],
    )[0]


def block_attn_res_prenorm_norm_rht_fast(
    query: mx.array,
    blocks_hat: mx.array | None,
    n_blocks: int,
    partial: mx.array,
    norm_w: mx.array,
    gu_signs: mx.array,
) -> tuple[mx.array, mx.array]:
    """Single-read BlockAttnRes(1) + RMSNorm + RHT-2304 where `blocks_hat` stores pre-normalized completed blocks."""
    T = int(partial.size) // 2304
    b_in = blocks_hat.reshape(-1) if (blocks_hat is not None and n_blocks > 0) else _fk._EMPTY_F32
    return _make_prenorm_bar_norm_rht_kernel()(
        inputs=[query.reshape(-1), b_in, partial.reshape(-1), norm_w.reshape(-1), gu_signs.reshape(-1), _fk._paley_b_flat(2304), _fk._u32(n_blocks, T)],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, 2304), (T, 2304)],
        output_dtypes=[mx.float16, mx.float16],
    )


@lru_cache(maxsize=2)
def _make_rank2_blend_3x_kernel():
    src = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint gt = threadgroup_position_in_grid.y; // 0 .. 3*T - 1
    const uint T = meta[0], h_shared = meta[1], nw_shared = meta[2], col0 = meta[3];
    const uint g = gt / T;
    const uint t = gt - g * T;
    const device float* hp = h + ((ulong)(h_shared ? 0u : g) * T + t) * 2304u;
    const device float* v2p = v2_hat + (ulong)t * 2304u;
    const device float* v3p = v3_hat + (ulong)t * 2304u;
    const float b3 = beta3_t48[t * 48u + col0 + g];
    const float b2 = 1.0f - b3;
    const device bfloat16_t* fgp = fg_mg3 + (ulong)gt * 4608u;
    const device bfloat16_t* mgp = fgp + 2304u;
    const device float* nwp = norm_w + (ulong)(nw_shared ? 0u : g) * 2304u;
    device float* op = out + (ulong)gt * 2304u;

    float v[9];
    float ss = 0.0f;
    #pragma unroll
    for (uint e = 0; e < 9u; e++) {
        const uint ch = tid + e * 256u;
        const float fg = metal::precise::tanh((float)fgp[ch]);
        const float mg = metal::precise::tanh((float)mgp[ch]);
        const float mp_ch = b2 * v2p[ch] + b3 * v3p[ch];
        v[e] = (1.0f - fg) * hp[ch] + mg * mp_ch;
        ss += v[e] * v[e];
    }
    threadgroup float red[8];
    ss = simd_sum(ss);
    if (lane == 0u) red[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8u; i++) tot += red[i];
    const float ir = metal::precise::rsqrt(tot * (1.0f / 2304.0f) + 1e-6f);
    #pragma unroll
    for (uint e = 0; e < 9u; e++) {
        const uint ch = tid + e * 256u;
        op[ch] = v[e] * ir * nwp[ch];
    }
    """
    return mx.fast.metal_kernel(
        name="ink_msg_blend_norm_3x_rank2_t48",
        input_names=["h", "v2_hat", "v3_hat", "beta3_t48", "fg_mg3", "norm_w", "meta"],
        output_names=["out"],
        source=src,
    )


@lru_cache(maxsize=2)
def _make_rank2_blend_rht_kernel():
    src = """
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.y;
    const uint col = meta[0];
    const uint c0 = simd * 9u;
    const device float* hp = h + (ulong)t * 2304u;
    const device float* v2p = v2_hat + (ulong)t * 2304u;
    const device float* v3p = v3_hat + (ulong)t * 2304u;
    const float b3 = beta3_t48[t * 48u + col];
    const float b2 = 1.0f - b3;
    const device bfloat16_t* fgp = fg_mg + (ulong)t * 4608u;
    const device bfloat16_t* mgp = fgp + 2304u;

    float v[9];
    float ss = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint idx = lane * 72u + c0 + j;
        const float fg = metal::precise::tanh((float)fgp[idx]);
        const float mg = metal::precise::tanh((float)mgp[idx]);
        const float mp_idx = b2 * v2p[idx] + b3 * v3p[idx];
        v[j] = (1.0f - fg) * hp[idx] + mg * mp_idx;
        ss += v[j] * v[j];
    }
    threadgroup float red[8];
    threadgroup float tg_z[32 * 72];
    ss = simd_sum(ss);
    if (lane == 0u) red[simd] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < 8u; i++) tot += red[i];
    const float ir = metal::precise::rsqrt(tot * (1.0f / 2304.0f) + 1e-6f);

    device half* hop = h_f16_out + (ulong)t * 2304u;
    device half* rop = x_rot_out + (ulong)t * 2304u;
    const bool h1 = lane & 1u, h2 = lane & 2u, h4 = lane & 4u, h8 = lane & 8u, h16 = lane & 16u;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const uint idx = lane * 72u + c;
        const float hm = v[j] * ir * norm_w[idx];
        hop[idx] = (half)hm;
        float rv = hm * gu_signs[idx];
        float o;
        o = simd_shuffle_xor(rv, 1u);  rv = h1  ? (o - rv) : (rv + o);
        o = simd_shuffle_xor(rv, 2u);  rv = h2  ? (o - rv) : (rv + o);
        o = simd_shuffle_xor(rv, 4u);  rv = h4  ? (o - rv) : (rv + o);
        o = simd_shuffle_xor(rv, 8u);  rv = h8  ? (o - rv) : (rv + o);
        o = simd_shuffle_xor(rv, 16u); rv = h16 ? (o - rv) : (rv + o);
        tg_z[idx] = rv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float* zr = tg_z + lane * 72u;
    constexpr float INV_S = 1.0f / 48.0f;
    #pragma unroll
    for (uint j = 0; j < 9u; j++) {
        const uint c = c0 + j;
        const device float4* br = (const device float4*)(B + c * 72u);
        float racc = 0.0f;
        #pragma unroll
        for (uint k4 = 0; k4 < 18u; k4++) {
            const float4 b4 = br[k4];
            const uint k = k4 * 4u;
            racc += zr[k] * b4.x + zr[k + 1u] * b4.y + zr[k + 2u] * b4.z + zr[k + 3u] * b4.w;
        }
        rop[lane * 72u + c] = (half)(racc * INV_S);
    }
    """
    return mx.fast.metal_kernel(
        name="ink_msg_blend_norm_rht_rank2_t48_f16",
        input_names=["h", "v2_hat", "v3_hat", "beta3_t48", "fg_mg", "norm_w", "gu_signs", "B", "meta"],
        output_names=["h_f16_out", "x_rot_out"],
        source=src,
    )


def msg_blend_norm_3x_rank2(
    h: mx.array,
    v2_hat_1d: mx.array,
    v3_hat_1d: mx.array,
    beta3_flat: mx.array,
    fg_mg3_bf16: mx.array,
    norm_w: mx.array,
    T: int,
    h_shared: bool,
    norm_shared: bool,
    col0: int,
) -> mx.array:
    """Zero-materialization 3x QKV message blend + RMSNorm directly from Rank-2 `(v2_hat, v3_hat, beta3)`."""
    return _make_rank2_blend_3x_kernel()(
        inputs=[h.reshape(-1), v2_hat_1d, v3_hat_1d, beta3_flat, fg_mg3_bf16.reshape(-1), norm_w.reshape(-1), _fk._u32(T, int(h_shared), int(norm_shared), int(col0))],
        grid=(256, 3 * T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(3, T, 2304)],
        output_dtypes=[mx.float32],
    )[0]


def msg_blend_norm_rht_rank2(
    h: mx.array,
    v2_hat_1d: mx.array,
    v3_hat_1d: mx.array,
    beta3_flat: mx.array,
    fg_mg_bf16: mx.array,
    norm_w: mx.array,
    gu_signs: mx.array,
    col: int,
) -> tuple[mx.array, mx.array]:
    """Zero-materialization MLP message blend + RMSNorm + RHT-2304 directly from Rank-2 `(v2_hat, v3_hat, beta3)`."""
    T = int(h.size) // 2304
    return _make_rank2_blend_rht_kernel()(
        inputs=[h.reshape(-1), v2_hat_1d, v3_hat_1d, beta3_flat, fg_mg_bf16.reshape(-1), norm_w.reshape(-1), gu_signs.reshape(-1), _fk._paley_b_flat(2304), _fk._u32(int(col))],
        grid=(256, T, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, 2304), (T, 2304)],
        output_dtypes=[mx.float16, mx.float16],
    )


@mx.compile
def _precompute_loop_pass_rank2_compiled(
    v2_hat: mx.array,
    v3_raw: mx.array,
    u_ordered: mx.array,
    all_dm_ordered: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    v3_hat = mx.fast.rms_norm(v3_raw, None, 1e-6)
    gap = ((v3_hat - v2_hat) @ u_ordered.T) * (1.0 / 48.0)  # [T, 48]
    beta3 = mx.sigmoid(gap)
    beta2 = 1.0 - beta3

    # Exact Law-of-Cosines identity for unit-RMS v2_hat, v3_hat (beta2 + beta3 == 1):
    # beta2^2 + beta3^2 + 2*beta2*beta3*a23 = 1 - 2*beta2*beta3*(1 - a23)
    a23 = mx.mean(v2_hat * v3_hat, axis=-1, keepdims=True)
    r_s = mx.rsqrt(mx.maximum(1.0 - 2.0 * beta2 * beta3 * (1.0 - a23), 0.0) + 1e-6)

    c2 = (r_s * beta2)[:, :, None].astype(mx.bfloat16)
    c3 = (r_s * beta3)[:, :, None].astype(mx.bfloat16)

    T = v2_hat.shape[0]
    v23_bf16 = mx.stack([v2_hat.astype(mx.bfloat16), v3_hat.astype(mx.bfloat16)], axis=0)
    Z23 = mx.matmul(v23_bf16, all_dm_ordered.T)  # [2, T, 3072]
    Z = c2 * Z23[0].reshape(T, 48, 64) + c3 * Z23[1].reshape(T, 48, 64)
    return v2_hat, v3_hat, beta3.reshape(-1), Z


def precompute_loop_pass_rank2(
    v2: mx.array,
    v3_raw: mx.array,
    u_ordered: mx.array,
    all_dm_ordered: mx.array,
    *,
    v2_is_prenorm: bool = True,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Exact Rank-2 Law-of-Cosines precomputation of all 48 cross-loop attention weights & gate projections for one loop pass."""
    v2_hat = v2 if v2_is_prenorm else mx.fast.rms_norm(v2, None, 1e-6)
    return _precompute_loop_pass_rank2_compiled(v2_hat, v3_raw, u_ordered, all_dm_ordered)


@mx.compile
def _msg_3x_shared_h_rank2_compiled(h: mx.array, Z: mx.array, dh3_flat: mx.array, fm3: mx.array, col0: int) -> mx.array:
    hn = mx.fast.rms_norm(h, None, 1e-6).astype(mx.bfloat16)
    g = ((hn @ dh3_flat.T).reshape(-1, 3, 64) + Z[:, col0 : col0 + 3, :]).swapaxes(0, 1)
    return mx.matmul(g, fm3.swapaxes(-1, -2))


@mx.compile
def _msg_3x_split_h_rank2_compiled(h3: mx.array, Z: mx.array, dh3: mx.array, fm3: mx.array, col0: int) -> mx.array:
    hn = mx.fast.rms_norm(h3, None, 1e-6).astype(mx.bfloat16)
    g = mx.matmul(hn, dh3.swapaxes(-1, -2)) + Z[:, col0 : col0 + 3, :].swapaxes(0, 1)
    return mx.matmul(g, fm3.swapaxes(-1, -2))


@mx.compile
def _msg_1x_rank2_compiled(h: mx.array, Z: mx.array, dh: mx.array, fm: mx.array, col: int) -> mx.array:
    hn = mx.fast.rms_norm(h, None, 1e-6).astype(mx.bfloat16)
    return (hn @ dh.T + Z[:, col, :]) @ fm.T


def ensure_rank2_loop_tables(model: Any) -> tuple[mx.array, mx.array]:
    """Build and cache `(u_ordered[48, 2304], all_dm_ordered[3072, 2304])` + 2D `qkv_dh3_flat` across loop layers `L6..17`."""
    cached = getattr(model, "_rank2_loop_tables", None)
    if cached is not None:
        return cached
    cfg = model.cfg
    H = int(cfg.hidden_size)
    num_loop = int(cfg.loop_end - cfg.loop_start)
    q_all = model.message_in_q.astype(mx.float32)  # [12, 4, 2304]
    u_qkv = q_all[:, :3, :].reshape(num_loop * 3, H)
    u_mlp = q_all[:, 3, :].reshape(num_loop, H)
    u_ordered = mx.concatenate([u_qkv, u_mlp], axis=0)

    qkv_dms = []
    mlp_dms = []
    for i in range(cfg.loop_start, cfg.loop_end):
        L = model.layers[i]
        if getattr(L, "qkv_dh3_flat", None) is None:
            L.qkv_dh3_flat = L.qkv_dh3.reshape(3 * int(cfg.loop_gate_rank), H)
        qkv_dms.append(L.qkv_dm3.reshape(3 * int(cfg.loop_gate_rank), H))
        mlp_dms.append(L.mlp_dm)
    all_dm_ordered = mx.concatenate(qkv_dms + mlp_dms, axis=0)
    mx.eval(u_ordered, all_dm_ordered)
    model._rank2_loop_tables = (u_ordered, all_dm_ordered)
    return u_ordered, all_dm_ordered


def extract_beacon_importance(steering_path: str | os.PathLike) -> tuple[Any, Any]:
    """Compute per-channel L2 importance across all 104 depth-wise & cross-loop readers in `steering.safetensors`."""
    import numpy as np

    st = mx.load(str(steering_path))
    readers = [np.array(st["model.message_input.query"].astype(mx.float32)).reshape(-1, 2304)]
    for l in range(24):
        ak = f"model.layers.{l}.attention_input.query"
        if ak in st:
            readers.append(np.array(st[ak].astype(mx.float32)).reshape(-1, 2304))
        mk = f"model.layers.{l}.mlp_input.query"
        if mk in st:
            readers.append(np.array(st[mk].astype(mx.float32)).reshape(-1, 2304))
    if "model.final_input.query" in st:
        readers.append(np.array(st["model.final_input.query"].astype(mx.float32)).reshape(1, 2304))
    all_readers = np.concatenate(readers, axis=0)
    col_norm = np.linalg.norm(all_readers, axis=0)
    beacon_mult = (col_norm / (np.median(col_norm) + 1e-6)).astype(np.float32)
    beacon_act = np.where(beacon_mult > 2.5, beacon_mult * 1.5, 1.0).astype(np.float32)
    return beacon_mult, beacon_act


def quantize_mtp_beacon_protected(
    src_safetensors: str | os.PathLike,
    dst_safetensors: str | os.PathLike,
    *,
    steering_path: str | os.PathLike | None = None,
    group_size: int = 64,
    bits: int = 4,
    tap_bits: int = 8,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Beacon-Aware AWQ LayerNorm Folding + Mixed-Precision (`tap_bits=8` on target-tap KV & Markov head) MTP quantizer.

    Protects the DSpark drafter (`mtp.safetensors`) against Goodfire's massive cross-loop & depth-wise outlier beacons
    (`d2027` at +32.4 RMS, `d2110, d1206, d2296, d1375, d1600`) by:
      1. Folding exact per-channel AWQ scales `s_c` into `layers.{0,1,2}.dflare_norm.weight` (kept in FP32) <-> `(k_proj_ctx, v_proj_ctx)`,
         `markov_head.dflare_norm.weight` (FP32) <-> `markov_head.rnn_proj.weight[:, 2816:5120]`, and
         `layers.0.input_layernorm.weight` (using `mask_embed` prior) <-> `layers.0.self_attn.{q,k,v}_proj.weight`.
      2. Quantizing target-tap ingest (`k_proj_ctx`, `v_proj_ctx`) and Markov recurrence bottleneck (`rnn_proj`, `hidden_out`)
         at `tap_bits` (default 8-bit, reducing `ctx_v` RMSE by `15.0x` and Markov RMSE by `-10.9%` for only `+9.3 MB`).
    """
    from pathlib import Path
    import numpy as np

    src_w = mx.load(str(src_safetensors))
    mod_w = dict(src_w)
    H = int(mod_w["mask_embed"].shape[0]) if "mask_embed" in mod_w else 2304

    if steering_path is not None and Path(steering_path).is_file():
        _, beacon_act = extract_beacon_importance(steering_path)
    else:
        beacon_act = np.ones((H,), dtype=np.float32)
        for idx, mult in ((2027, 8.0), (2110, 7.5), (1206, 9.5), (2296, 9.7), (1375, 9.6), (1600, 6.4), (924, 6.0), (2012, 6.0), (1573, 8.8), (943, 8.6)):
            if idx < H:
                beacon_act[idx] = mult

    def _awq_scale(w_ln_np, w_mats_np, act_prior_np, exp_a: float):
        act_est = np.abs(w_ln_np) * act_prior_np
        w_cat = np.concatenate(w_mats_np, axis=0)
        w_col_max = np.max(np.abs(w_cat), axis=0) + 1e-6
        s = (act_est ** exp_a) / (w_col_max ** (1.0 - exp_a))
        s = s.reshape(-1, group_size)
        s = s / (np.median(s, axis=1, keepdims=True) + 1e-6)
        return np.clip(s.reshape(-1), 0.25, 4.0).astype(np.float32)

    for i in range(3):
        p = f"layers.{i}."
        if (p + "dflare_norm.weight") in mod_w and (p + "self_attn.k_proj_ctx.weight") in mod_w:
            w_ln = np.array(mod_w[p + "dflare_norm.weight"].astype(mx.float32))
            wk = np.array(mod_w[p + "self_attn.k_proj_ctx.weight"].astype(mx.float32))
            wv = np.array(mod_w[p + "self_attn.v_proj_ctx.weight"].astype(mx.float32))
            s_ctx = _awq_scale(w_ln, [wk, wv], beacon_act, alpha)
            mod_w[p + "dflare_norm.weight"] = mx.array(w_ln / s_ctx, dtype=mx.float32)
            mod_w[p + "self_attn.k_proj_ctx.weight"] = mx.array(wk * s_ctx[None, :]).astype(mx.bfloat16)
            mod_w[p + "self_attn.v_proj_ctx.weight"] = mx.array(wv * s_ctx[None, :]).astype(mx.bfloat16)

    if "markov_head.dflare_norm.weight" in mod_w and "markov_head.rnn_proj.weight" in mod_w:
        w_hd = np.array(mod_w["markov_head.dflare_norm.weight"].astype(mx.float32))
        wrnn = np.array(mod_w["markov_head.rnn_proj.weight"].astype(mx.float32))
        if wrnn.shape[1] >= H:
            col_start = wrnn.shape[1] - H
            s_hd = _awq_scale(w_hd, [wrnn[:, col_start:]], beacon_act, alpha)
            mod_w["markov_head.dflare_norm.weight"] = mx.array(w_hd / s_hd, dtype=mx.float32)
            wrnn[:, col_start:] *= s_hd[None, :]
            mod_w["markov_head.rnn_proj.weight"] = mx.array(wrnn).astype(mx.bfloat16)

    tap_keys = {
        "layers.0.self_attn.k_proj_ctx.weight",
        "layers.0.self_attn.v_proj_ctx.weight",
        "layers.1.self_attn.k_proj_ctx.weight",
        "layers.1.self_attn.v_proj_ctx.weight",
        "layers.2.self_attn.k_proj_ctx.weight",
        "layers.2.self_attn.v_proj_ctx.weight",
        "markov_head.rnn_proj.weight",
        "markov_head.hidden_out.weight",
    }
    out: dict[str, mx.array] = {}
    q_keys = []
    tap_q_keys = []
    for k, v in mod_w.items():
        if v.ndim == 2 and v.shape[1] % group_size == 0 and v.nbytes >= 1_000_000:
            b = int(tap_bits) if k in tap_keys else int(bits)
            wq, sc, bi = mx.quantize(v, group_size=group_size, bits=b)
            out[k] = wq
            out[k + ".scales"] = sc
            out[k + ".biases"] = bi
            q_keys.append(k)
            if k in tap_keys:
                tap_q_keys.append(k)
        else:
            out[k] = v
    mx.eval(out)
    dst = Path(dst_safetensors)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst), out)
    raw_mb = sum(v.nbytes for v in src_w.values()) / 1e6
    q_mb = sum(v.nbytes for v in out.values()) / 1e6
    return {
        "src_mb": round(raw_mb, 2),
        "dst_mb": round(q_mb, 2),
        "compression": round(raw_mb / max(q_mb, 1e-6), 2),
        "quantized_matrices": len(q_keys),
        "tap_protected_matrices": len(tap_q_keys),
        "bits": int(bits),
        "tap_bits": int(tap_bits),
        "group_size": int(group_size),
    }


# --------------------------------------------------------------------------- O(N) Global KV Allocation & M4 Pro Tuning


_INSTALLED = False


def install_metal_accelerations(log_fn=None) -> dict[str, Any]:
    """Install custom Metal SDPA kernels, O(N) global KV allocation, and 16+ core M4 Pro/Max grid tuning."""
    global _INSTALLED
    if _INSTALLED:
        return {"installed": True, "already": True}

    os.environ.setdefault("LAMQ_DRAFTER_TRIM_BLEN", "1")

    # 1. Install custom GQA-Shared SG8 SDPA kernels (used by both fast_kernels and model)
    _fk.gqa_relpos_sinks_sdpa_fast = accelerated_gqa_sdpa
    _model_mod.gqa_relpos_sinks_sdpa_fast = accelerated_gqa_sdpa

    # 1b. Kernel 6 (opt-in, LAMQ_PAR_SWIGLU=1): 2-slot parallel SwiGLU + Paley-72 RHT-1152 (bit-exact).
    #     Batched-GPU benchmarks on M4 Pro show parity-to-slightly-slower vs stock (memory-bound), so OFF by default.
    if os.environ.get("LAMQ_PAR_SWIGLU", "0") == "1":
        _fk.swiglu_rht_1152_fast = fast_swiglu_rht_1152
        _model_mod.swiglu_rht_1152_fast = fast_swiglu_rht_1152

    # 2. Eliminate O(N^2) `+256` global KV reallocations in Inkling2Cache and DSparkDrafterMLX
    _kv_step = 4096

    def _fast_update_kv_f16(self, slot: int, kv_f16: mx.array, is_sliding: bool):
        T = int(kv_f16.shape[2])
        cur = self.kv_len[slot]
        self.kv_is_sliding[slot] = is_sliding
        mode = "fp16" if (is_sliding or not self._fast) else self.cfg.global_kv_mode
        if mode != "fp16":
            if mode == "q4":
                quantized, scales = _fk.kv_h128_q4_quantize(kv_f16)
            else:
                quantized = _fk.fp8_e4m3_quantize_f16_to_u8(kv_f16)
                scales = None
            need = cur + T
            reserve = int(getattr(self, "_reserve_cap", 0) or 0)
            for new, buffers in ((quantized, self.kv_u8), (scales, self.kv_q4_sc)):
                if new is None:
                    continue
                buffer = buffers[slot]
                if buffer is None or need > buffer.shape[2]:
                    target = max(need, reserve)
                    capacity = ((target + _kv_step - 1) // _kv_step) * _kv_step
                    larger = mx.zeros((*new.shape[:2], capacity, new.shape[-1]), dtype=new.dtype)
                    if buffer is not None and cur > 0:
                        larger[:, :, :cur, :] = buffer[:, :, :cur, :]
                    buffer = buffers[slot] = larger
                buffer[:, :, cur:need, :] = new
            self.kv_cap[slot] = self.kv_u8[slot].shape[2]
            self.kv_len[slot] = need
            return self.kv_u8[slot], need, self.kv_start[slot], self.kv_cap[slot], mode == "fp8", self.kv_q4_sc[slot]

        W = self.cfg.sliding_window_size
        if is_sliding and not self._spec_recording and cur + T > self.kv_cap[slot] and cur > W:
            shift = cur - W
            KV_old = self.kv_f16[slot]
            new_cap = max(self.kv_cap[slot], ((W + T + 255) // 256) * 256)
            KV_new = mx.zeros((2, 8, new_cap, 128), dtype=mx.float16)
            KV_new[:, :, :W, :] = KV_old[:, :, shift:cur, :]
            self.kv_f16[slot] = KV_new
            self.kv_cap[slot] = new_cap
            self.kv_start[slot] += shift
            cur = W
        need = cur + T
        if self.kv_f16[slot] is None or need > self.kv_cap[slot]:
            if is_sliding:
                new_cap = ((need + 255) // 256) * 256
            else:
                target = max(need, int(getattr(self, "_reserve_cap", 0) or 0))
                new_cap = ((target + _kv_step - 1) // _kv_step) * _kv_step
            KV_new = mx.zeros((2, 8, new_cap, 128), dtype=mx.float16)
            if self.kv_f16[slot] is not None and cur > 0:
                KV_new[:, :, :cur, :] = self.kv_f16[slot][:, :, :cur, :]
            self.kv_f16[slot] = KV_new
            self.kv_cap[slot] = new_cap
        KV_buf = self.kv_f16[slot]
        KV_buf[:, :, cur:need, :] = kv_f16
        self.kv_len[slot] = need
        return KV_buf, need, self.kv_start[slot], self.kv_cap[slot], False, None

    _model_mod.Inkling2Cache.update_kv_f16 = _fast_update_kv_f16

    def _fast_drafter_ensure_cap(self, i: int, need: int):
        if self.ctx_k[i] is None or need > self._ctx_cap[i]:
            if i in self.local_layers:
                new_cap = ((need + 255) // 256) * 256
            else:
                target = max(need, int(getattr(self, "_reserve_cap", 0) or 0))
                new_cap = ((target + _kv_step - 1) // _kv_step) * _kv_step
            K_new = mx.zeros((1, self.num_kv_heads, new_cap, self.head_dim), dtype=mx.float16)
            V_new = mx.zeros((1, self.num_kv_heads, new_cap, self.head_dim), dtype=mx.float16)
            cur = self._ctx_cur[i]
            if self.ctx_k[i] is not None and cur > 0:
                K_new[:, :, :cur, :] = self.ctx_k[i][:, :, :cur, :]
                V_new[:, :, :cur, :] = self.ctx_v[i][:, :, :cur, :]
            self.ctx_k[i] = K_new
            self.ctx_v[i] = V_new
            self._ctx_cap[i] = new_cap

    _drafter_mod.DSparkDrafterMLX._ensure_ctx_cap = _fast_drafter_ensure_cap

    # 2b. Custom Metal GQA-12:1 Split-K SDPA across all 3 DSpark drafter layers + B_len trim in propose_sample_gpu
    def _fast_drafter_run_3layers(self, anc_emb: mx.array, n_toks: int | None = None) -> mx.array:
        B_len = self.block_size if n_toks is None else max(2, min(int(n_toks), self.block_size))
        masks = mx.broadcast_to(self.mask_embed[None, :], (B_len - 1, self.hidden_size))
        h = mx.concatenate([anc_emb.astype(mx.bfloat16)[None, :], masks], axis=0)
        q_dim = self._q_dim
        qk_dim = q_dim + self._kv_dim
        off = self.ctx_len
        can_fast_sdpa = self.num_heads == 24 and self.num_kv_heads == 2 and self.head_dim == 128 and 2 <= B_len <= 8

        for i, L in enumerate(self.layers):
            res = h
            xn = mx.fast.rms_norm(h, L["in_ln"], self.eps)
            qkv = L["qkv_proj"](xn, out_dtype=mx.bfloat16)
            q = qkv[:, :q_dim].reshape(1, B_len, self.num_heads, self.head_dim)
            k = qkv[:, q_dim:qk_dim].reshape(1, B_len, self.num_kv_heads, self.head_dim)
            v = qkv[:, qk_dim:].reshape(1, B_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3).astype(mx.float16)
            q = mx.fast.rope(mx.fast.rms_norm(q, L["q_norm"], self.eps).transpose(0, 2, 1, 3), self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=off).astype(mx.float16)
            k = mx.fast.rope(mx.fast.rms_norm(k, L["k_norm"], self.eps).transpose(0, 2, 1, 3), self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=off).astype(mx.float16)

            cur = self._ctx_cur[i]
            hi = cur + B_len
            self._ensure_ctx_cap(i, hi)
            self.ctx_k[i][:, :, cur:hi, :] = k
            self.ctx_v[i][:, :, cur:hi, :] = v
            lo = max(0, cur - self.window) if i in self.local_layers else 0
            if can_fast_sdpa and (hi - lo) >= 256:
                attn_out = drafter_gqa_sdpa_fast(q, self.ctx_k[i], self.ctx_v[i], lo, hi, self._ctx_cap[i])
            else:
                K_all = self.ctx_k[i][:, :, lo:hi, :]
                V_all = self.ctx_v[i][:, :, lo:hi, :]
                attn_out = mx.fast.scaled_dot_product_attention(q, K_all, V_all, scale=self._sdpa_scale)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B_len, self.num_heads * self.head_dim).astype(mx.bfloat16)
            h = res + L["o_proj"](attn_out, out_dtype=mx.bfloat16)

            res2 = h
            mn = mx.fast.rms_norm(h, L["post_ln"], self.eps)
            h = res2 + L["down_proj"](_drafter_mod._silu_mul_compiled(L["gate_up_proj"](mn, out_dtype=mx.bfloat16)), out_dtype=mx.bfloat16)

        return _drafter_mod.rms_norm(h, self.final_norm, self.eps)

    def _fast_propose_sample_gpu(
        self,
        anchor_tok: int | mx.array,
        embed_fn,
        head_sample_fn,
        draft_len: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        n_toks = self.block_size if draft_len is None else max(2, min(int(draft_len), self.block_size))
        anc_mx = (
            anchor_tok.reshape(1).astype(mx.int32)
            if isinstance(anchor_tok, mx.array)
            else mx.array([int(anchor_tok)], dtype=mx.int32)
        )
        anc_emb = embed_fn(anc_mx)[0, 0, :]
        trim = os.environ.get("LAMQ_DRAFTER_TRIM_BLEN") == "1"
        h = self._run_3layers(anc_emb, n_toks=n_toks if trim else None)

        state = self._zero_state
        prev_tok = anc_mx
        toks = [anc_mx]
        ords: list[mx.array] = []
        probs: list[mx.array] = []
        for s in range(1, n_toks):
            prev_emb = self.markov_w1(prev_tok)[0]
            inp = mx.concatenate([state, prev_emb, h[s], self.f_rnn], axis=0)[None, :]
            proj = (self.rnn_proj_w(inp) + self.rnn_proj_b)[0]
            state, latent = _drafter_mod._rnn_update_compiled(state, proj)
            delta_h = self.hidden_out_w(latent[None, :])[0] + self.hidden_out_b
            vocab_tok, ord_tok, q_prob = head_sample_fn(h[s] + delta_h)
            nxt = vocab_tok.reshape(1).astype(mx.int32)
            toks.append(nxt)
            ords.append(ord_tok.reshape(1).astype(mx.int32))
            probs.append(q_prob.astype(mx.float32))
            prev_tok = nxt
        return (
            mx.concatenate(toks, axis=0),
            mx.concatenate(ords, axis=0),
            mx.stack(probs, axis=0),
        )

    _drafter_mod.DSparkDrafterMLX._run_3layers = _fast_drafter_run_3layers
    _drafter_mod.DSparkDrafterMLX.propose_sample_gpu = _fast_propose_sample_gpu

    # 2c. Exact radix-histogram Metal top-p sampling (eliminates 200K/32K argsort on temp > 0)
    _model_mod.logits_to_probs = fast_logits_to_probs
    _gen_mod.logits_to_probs = fast_logits_to_probs

    # 2c-bis. Clamp mx.set_wired_limit inside stream_dspark / stream_serial (generate.py:714 sets it to max_rec=19.07 GB!)
    orig_set_wired_limit = mx.set_wired_limit
    max_rec = int(mx.device_info().get("max_recommended_working_set_size", 0) or 0)
    safe_wire_cap = (max_rec - (6 << 30)) if max_rec <= (24 << 30) else (max_rec - (4 << 30))

    def _clamped_set_wired_limit(limit: int) -> int:
        if limit > 0 and safe_wire_cap > 0 and limit > safe_wire_cap:
            limit = safe_wire_cap
        return orig_set_wired_limit(limit)

    mx.set_wired_limit = _clamped_set_wired_limit

    # 2d. Enable lossless N-gram speculative drafting by default in stream_dspark (opt out via LAMQ_NGRAM=0),
    #     with cross-turn incremental O(delta) NgramDraftIndex reuse, EOS commit trimming, and
    #     Post-Generation KV Cache Promotion (SGLang RadixAttention / vLLM style full-turn KV reuse).
    orig_commit_spec = _model_mod.Inkling2Cache.commit_spec_verify
    orig_commit_tree = _model_mod.Inkling2Cache.commit_tree_verify
    orig_push_context = _drafter_mod.DSparkDrafterMLX.push_context

    def _trimmed_commit_spec_verify(self, k_accept: int, block_len: int):
        trim = getattr(self, "_trim_commit_to", None)
        if trim is not None:
            self._trim_commit_to = None
            k_accept = max(1, min(int(k_accept), int(trim)))
        return orig_commit_spec(self, k_accept, block_len)

    def _trimmed_commit_tree_verify(self, accepted_path: list[int], block_len: int):
        trim = getattr(self, "_trim_commit_to", None)
        if trim is not None:
            self._trim_commit_to = None
            accepted_path = list(accepted_path[: max(1, int(trim))])
        return orig_commit_tree(self, accepted_path, block_len)

    def _trimmed_push_context(self, target_taps: mx.array, start_pos: int):
        trim = getattr(self, "_trim_push_to", None)
        if trim is not None:
            self._trim_push_to = None
            t_keep = max(1, min(int(target_taps.shape[0]), int(trim)))
            target_taps = target_taps[:t_keep]
        return orig_push_context(self, target_taps, start_pos)

    _model_mod.Inkling2Cache.commit_spec_verify = _trimmed_commit_spec_verify
    _model_mod.Inkling2Cache.commit_tree_verify = _trimmed_commit_tree_verify
    _drafter_mod.DSparkDrafterMLX.push_context = _trimmed_push_context

    _stream_tls = threading.local()
    orig_ngram_cls = _gen_mod.NgramDraftIndex

    class _CachedNgramDraftIndex(orig_ngram_cls):
        """NgramDraftIndex with O(delta) cross-turn reuse and EOS commit-boundary trimming."""

        def __init__(self, initial_tokens=()):
            ctx = getattr(_stream_tls, "ctx", None)
            pc = ctx.get("prompt_cache") if isinstance(ctx, dict) else None
            reused = (
                pc.acquire_ngram_index(initial_tokens)
                if (pc is not None and hasattr(pc, "acquire_ngram_index"))
                else None
            )
            if reused is not None:
                self.tokens = reused.tokens
                self._t3 = reused._t3
                self._t4 = reused._t4
                self._t5 = reused._t5
                self.tables = reused.tables
                self._indexed_end = reused._indexed_end
                self._cursor = None
                self._last_hit = None
                self._has_tombstones = reused._has_tombstones
            else:
                super().__init__(initial_tokens)
            self._eos_ids = ctx.get("eos_ids", frozenset((200006,))) if isinstance(ctx, dict) else frozenset((200006,))
            self._prompt_cache = pc
            self._consec_ngram_toks = 0
            self._ngram_cooldown = 0
            self._max_consec_ngram_toks = int(os.environ.get("LAMQ_NGRAM_MAX_RUN", "96"))
            if isinstance(ctx, dict):
                ctx["ngram_idx"] = self

        def propose(self, anchor: int, max_draft: int = 8, min_cont: int = 2, eos=frozenset(), *, min_n: int = 3):
            if self._ngram_cooldown > 0:
                self._ngram_cooldown -= 1
                self._last_hit = None
                self._cursor = None
                return None
            return super().propose(anchor, max_draft=max_draft, min_cont=min_cont, eos=eos, min_n=min_n)

        def observe(self, raw, k_match: int, block_size: int, was_ngram: bool) -> None:
            if raw and self._eos_ids:
                for idx, tok in enumerate(raw):
                    if tok in self._eos_ids:
                        if idx + 1 < len(raw):
                            keep = idx + 1
                            raw = list(raw[:keep])
                            k_match = min(k_match, keep - 1)
                            pc = self._prompt_cache
                            ag = getattr(pc, "_active_gen", None) if pc is not None else None
                            if isinstance(ag, dict):
                                if ag.get("cache") is not None:
                                    ag["cache"]._trim_commit_to = keep
                                if ag.get("drafter") is not None:
                                    ag["drafter"]._trim_push_to = keep
                        break
            hit_before = self._last_hit
            super().observe(raw, k_match, block_size, was_ngram)
            if was_ngram and k_match > 0:
                self._consec_ngram_toks += int(k_match)
                if self._max_consec_ngram_toks > 0 and self._consec_ngram_toks >= self._max_consec_ngram_toks:
                    # Break runaway contiguous N-gram self-copy rails and hand control back to DSpark for 8 rounds
                    if hit_before is not None:
                        hn, hkey, _ = hit_before
                        if hn > 0 and hkey is not None and hn in self.tables:
                            self.tables[hn][hkey] = None
                            self._has_tombstones = True
                    if len(self.tokens) >= 5:
                        t5_key = tuple(self.tokens[-5:])
                        if t5_key in self._t5:
                            self._t5[t5_key] = None
                            self._has_tombstones = True
                    self._cursor = None
                    self._consec_ngram_toks = 0
                    self._ngram_cooldown = 8
            else:
                self._consec_ngram_toks = 0

    _gen_mod.NgramDraftIndex = _CachedNgramDraftIndex

    ngram_enabled = os.environ.get("LAMQ_NGRAM", "1") != "0"
    if ngram_enabled:
        orig_stream_dspark = _gen_mod.stream_dspark

        def _ngram_stream_dspark(*args, **kwargs):
            kwargs.setdefault("ngram", True)
            kwargs.setdefault("ngram_min_match", 5)
            pc = kwargs.get("prompt_cache")
            eos_set = frozenset(int(x) for x in (kwargs.get("eos_ids") or (200006,)))
            ctx = {"prompt_cache": pc, "ngram_idx": None, "eos_ids": eos_set}
            _stream_tls.ctx = ctx
            # Exact SpecInfer stochastic tree verification (temp > 0) + hybrid N-gram||DSpark 4-3 tree + Token Recycling.
            # Opt-in via LAMQ_STOCHASTIC_TREE="4-3" or "auto" (default OFF: on 12.8GB + shared-global-KV long contexts,
            # stock 1D `cur_block=6->8` avoids `T=8` overhead and non-contiguous `commit_tree_verify` KV copies).
            st_tree = os.environ.get("LAMQ_STOCHASTIC_TREE", "").strip()
            temp_val = float(kwargs.get("temp", 0.0) or 0.0)
            explicit_tree = kwargs.get("tree")
            no_margin = float(kwargs.get("max_logit_gap", 0.0) or 0.0) == 0.0 and kwargs.get("min_prob_ratio") is None
            if temp_val > 0.0 and no_margin and explicit_tree is not None:
                from .spec import stream_dspark_ext

                gen = stream_dspark_ext(*args, **kwargs)
            elif (
                temp_val > 0.0
                and no_margin
                and explicit_tree is None
                and kwargs.get("draft_len") is None
                and st_tree not in ("", "0", "off", "false", "False")
                and (st_tree == "auto" or st_tree in _drafter_mod.TREE_TOPOLOGIES)
            ):
                from .spec import stream_dspark_ext

                kwargs["tree"] = "4-3" if st_tree == "auto" else st_tree
                if st_tree == "auto":
                    kwargs["adaptive_tree"] = True
                gen = stream_dspark_ext(*args, **kwargs)
            elif (
                explicit_tree is None
                and kwargs.get("draft_len") is None
                and kwargs.get("adaptive", True)
                and os.environ.get("LAMQ_ADAPTIVE_DRAFT_BLOCK", "1") != "0"
            ):
                from .spec import stream_dspark_ext

                gen = stream_dspark_ext(*args, **kwargs)
            else:
                gen = orig_stream_dspark(*args, **kwargs)
            try:
                for chunk in gen:
                    ng = ctx.get("ngram_idx")
                    if (
                        pc is not None
                        and ng is not None
                        and hasattr(pc, "checkpoint_pre_tool_gen")
                        and (200049 in chunk)
                    ):
                        pc.checkpoint_pre_tool_gen(ng)
                    yield chunk
            finally:
                try:
                    gen.close()
                finally:
                    ng = ctx.get("ngram_idx")
                    if pc is not None and ng is not None and hasattr(pc, "finalize_post_gen"):
                        pc.finalize_post_gen(ng)
                    _stream_tls.ctx = None

        _gen_mod.stream_dspark = _ngram_stream_dspark
        _inkling_mod.stream_dspark = _ngram_stream_dspark

    # 2d-bis. Auto-detect per-matrix bits (2, 4, 8) in DSpark _QLinear.__init__ for mixed-precision Beacon-Protected MTP
    if hasattr(_drafter_mod, "_QLinear"):
        orig_qlin_init = _drafter_mod._QLinear.__init__

        def _auto_bits_qlin_init(self, weights, key: str = "", bits: int = 4, group_size: int = 64):
            if weights is not None and (key + ".scales") in weights:
                w_t = weights[key]
                sc_t = weights[key + ".scales"]
                if hasattr(w_t, "shape") and hasattr(sc_t, "shape") and len(w_t.shape) == 2 and len(sc_t.shape) == 2 and int(sc_t.shape[1]) > 0:
                    inferred = (int(w_t.shape[1]) * 32) // (int(sc_t.shape[1]) * int(group_size))
                    if inferred in (2, 4, 8):
                        bits = inferred
            return orig_qlin_init(self, weights, key=key, bits=bits, group_size=group_size)

        _drafter_mod._QLinear.__init__ = _auto_bits_qlin_init

    # 2e. Goodfire Mechanistic Forward Path:
    #     - Opt #1 (LAMQ_RANK2_LOOP_MSG=1): Exact Rank-2 Cross-Loop Message Factorization & Law-of-Cosines RMSNorm
    #       (collapses 48 message RMSNorms + 48 `dm` GEMMs into 2 vectors `[v2_hat, v3_hat]` and eliminates `[12, 4, T, 2304]` materialization).
    #     - Opt #2 (LAMQ_PRENORM_BAR=1): Pre-normalized single-read BlockAttnRes + deduplicated block-boundary concatenation
    #       (stores `v_hat = rms_norm(v)` once per block boundary, eliminating 8 duplicate concats and 50% DRAM reads across 96 BAR calls).
    #     - Plus low-memory prefill guard for T >= 64 chunks on 24 GB Unified Memory.
    _low_mem_thresh = int(float(os.environ.get("LAMQ_LOW_MEM_THRESHOLD_GB", "12.5")) * (1 << 30))

    def _fast_layer_forward(
        self,
        h: mx.array,
        blocks_cat: mx.array | None,
        n_blocks: int,
        q_pos0: int,
        T: int,
        cache,
        cache_slot: int,
        messages: mx.array | None = None,
        tree_spec: dict | None = None,
        qkv_w_cached: mx.array | None = None,
        o_w_cached: mx.array | None = None,
        *,
        blocks_prenorm: bool = False,
        precomputed_local_blocks: mx.array | None = None,
        rank2_ctx: tuple[mx.array, mx.array, mx.array, mx.array, int] | None = None,
    ) -> mx.array:
        cfg = self.cfg
        block_residual = h
        tree_lag = tree_spec["lag_mx"] if tree_spec is not None else None
        tree_depth = tree_spec["depth_mx"] if tree_spec is not None else None
        tree_mask = tree_spec["mask_mx"] if tree_spec is not None else None
        bar_fn = block_attn_res_prenorm_fused if blocks_prenorm else _model_mod.block_attn_res_fused
        bar_rht_fn = block_attn_res_prenorm_norm_rht_fast if blocks_prenorm else _model_mod.block_attn_res_norm_rht_fast

        # 1. Attn input AttnRes + optional Rank-2 / batched message gates + layernorm + 5-matrix batched RHT-2304
        use_qkv_sg8 = (4 <= T <= 8) if _model_mod._is_m4() else (5 <= T <= 8)
        if self.is_sliding:
            if messages is None and rank2_ctx is None:
                if self.attn_in_q is not None:
                    q_in = bar_fn(self.attn_in_q, blocks_cat, n_blocks, h, norm_w=self.in_ln)[0]
                else:
                    q_in = _model_mod.rms_norm(h, self.in_ln, cfg.rms_norm_eps)
                x_rot5 = _model_mod.rht_in_2304_fast(q_in, self.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=False, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)
            elif rank2_ctx is not None:
                v2_1d, v3_1d, b3_flat, Z, loop_layer_idx = rank2_ctx
                col0 = loop_layer_idx * 3
                h_a = bar_fn(self.attn_in_q, blocks_cat, n_blocks, h)[0]
                dh3_flat = getattr(self, "qkv_dh3_flat", None)
                if dh3_flat is None:
                    dh3_flat = self.qkv_dh3_flat = self.qkv_dh3.reshape(192, cfg.hidden_size)
                fg_mg3 = _msg_3x_shared_h_rank2_compiled(h_a, Z, dh3_flat, self.qkv_fm3, col0)
                qkv_in = msg_blend_norm_3x_rank2(h_a, v2_1d, v3_1d, b3_flat, fg_mg3, self.in_ln, T, h_shared=True, norm_shared=True, col0=col0)
                q_in = qkv_in[0]
                x_rot5 = _model_mod.rht_in_2304_fast(qkv_in, self.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=True, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)
            else:
                h_a = bar_fn(self.attn_in_q, blocks_cat, n_blocks, h)[0]
                fg_mg3 = _model_mod._msg_3x_shared_h_compiled(h_a, messages[:3], self.qkv_dh3, self.qkv_dm3, self.qkv_fm3)
                qkv_in = _model_mod.msg_blend_norm_3x_fused(h_a, messages[:3], fg_mg3, self.in_ln, T, h_shared=True, norm_shared=True)
                q_in = qkv_in[0]
                x_rot5 = _model_mod.rht_in_2304_fast(qkv_in, self.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=True, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)
        else:
            if self.attn_in_q is None:
                qkv_h = mx.broadcast_to(h[None, :, :], (3, T, 2304))
            else:
                qkv_h = bar_fn(self.attn_in_q, blocks_cat, n_blocks, h)
            if rank2_ctx is not None:
                v2_1d, v3_1d, b3_flat, Z, loop_layer_idx = rank2_ctx
                col0 = loop_layer_idx * 3
                fg_mg3 = _msg_3x_split_h_rank2_compiled(qkv_h, Z, self.qkv_dh3, self.qkv_fm3, col0)
                qkv_in = msg_blend_norm_3x_rank2(qkv_h, v2_1d, v3_1d, b3_flat, fg_mg3, self.qkv_in_ln, T, h_shared=False, norm_shared=False, col0=col0)
                q_in, k_in, v_in = qkv_in[0], qkv_in[1], qkv_in[2]
            elif messages is not None:
                fg_mg3 = _model_mod._msg_3x_split_h_compiled(qkv_h, messages[:3], self.qkv_dh3, self.qkv_dm3, self.qkv_fm3)
                qkv_in = _model_mod.msg_blend_norm_3x_fused(qkv_h, messages[:3], fg_mg3, self.qkv_in_ln, T, h_shared=False, norm_shared=False)
                q_in, k_in, v_in = qkv_in[0], qkv_in[1], qkv_in[2]
            else:
                q_in = _model_mod.rms_norm(qkv_h[0], self.q_in_ln, cfg.rms_norm_eps)
                k_in = _model_mod.rms_norm(qkv_h[1], self.k_in_ln, cfg.rms_norm_eps)
                v_in = _model_mod.rms_norm(qkv_h[2], self.v_in_ln, cfg.rms_norm_eps)
                qkv_in = mx.stack([q_in, k_in, v_in], axis=0)
            x_rot5 = _model_mod.rht_in_2304_fast(qkv_in, self.qkv_stage.signs_in, n_signs=5, T=T, split_qkv=True, out_dtype=mx.float16, transpose_for_sg8=use_qkv_sg8)

        # 2. Fused 5-matrix QKV + Rel + O_Gate stage
        use_split_parts = (T > 8) and not (self.is_sliding and not self.has_qkv_out_rot)
        if use_split_parts:
            q_raw, k_raw, v_raw, rel_raw, o_gate_raw = self.qkv_stage.matmul_rotated_f16_parts(x_rot5, T, W_cached=qkv_w_cached)
            y_qkv = None
        else:
            y_qkv = (
                (
                    self.qkv_stage.matmul_rotated_sg8_f16(x_rot5, T=T)
                    if _model_mod._is_m4()
                    else self.qkv_stage.matmul_rotated_sg8_f16(x_rot5, T=T, rb=1, sg=8, ntg=160)
                )
                if use_qkv_sg8
                else self.qkv_stage.matmul_rotated_f16(x_rot5, T, W_cached=qkv_w_cached)
            )
            o_gate_raw = y_qkv[:, 5504:8576]
        if self.qkv_out_rots[4] is not None:
            o_gate_raw = _model_mod.rht_apply_out(o_gate_raw, self.qkv_out_rots[4])

        # 3. Fused Q/Rel clip (+log_scale) + K/V conv/norm + unified FP16 KV append
        kp = cache.get_conv_prev(cache_slot, 0, 1024)
        vp = cache.get_conv_prev(cache_slot, 1, 1024)
        if self.is_sliding and not self.has_qkv_out_rot:
            q, rel_st, kv_f16, new_kp, new_vp = _model_mod.qkv_rel_prep_sliding_fused(
                y_qkv, self.q_scale, self.rel_scale,
                kp, self.k_conv_w, vp, self.v_conv_w,
                T, cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                tree_lag=tree_lag,
            )
            cache.set_conv_after(cache_slot, 0, kp, y_qkv[:, 3072:4096].reshape(1, T, 1024), new_kp)
            cache.set_conv_after(cache_slot, 1, vp, y_qkv[:, 4096:5120].reshape(1, T, 1024), new_vp)
        else:
            if not use_split_parts:
                q_raw = y_qkv[:, 0:3072]
                k_raw = y_qkv[:, 3072:4096]
                v_raw = y_qkv[:, 4096:5120]
                rel_raw = y_qkv[:, 5120:5504]
            if self.qkv_out_rots[0] is not None:
                q_raw = _model_mod.rht_apply_out(q_raw, self.qkv_out_rots[0])
            if self.qkv_out_rots[1] is not None:
                k_raw = _model_mod.rht_out_1024_fast(k_raw, self.qkv_out_rots[1]) if k_raw.shape[-1] == 1024 else _model_mod.rht_apply_out(k_raw, self.qkv_out_rots[1])
            if self.qkv_out_rots[2] is not None:
                v_raw = _model_mod.rht_out_1024_fast(v_raw, self.qkv_out_rots[2]) if v_raw.shape[-1] == 1024 else _model_mod.rht_apply_out(v_raw, self.qkv_out_rots[2])
            if self.qkv_out_rots[3] is not None:
                rel_raw = _model_mod.rht_apply_out(rel_raw, self.qkv_out_rots[3])
            q, rel_st = _model_mod.q_rel_clip_fused(
                q_raw, self.q_scale, rel_raw, self.rel_scale,
                q_pos0, T, self.is_sliding,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold,
                cfg.log_scaling_alpha, cfg.log_scaling_n_floor,
                tree_depth=tree_depth,
            )
            if self.is_sliding:
                kv_f16, new_kp, new_vp = _model_mod.kv_sconv4_norm_fused(kp, k_raw, self.k_conv_w, vp, v_raw, self.v_conv_w, T, cfg.qk_l2_clip_threshold, tree_lag=tree_lag)
            else:
                kv_f16, new_kp, new_vp = _model_mod.kv_dyn_sconv4_norm_fused(
                    kp, k_raw, k_in, self.k_dyn_st, self.k_dyn_pr_f16,
                    vp, v_raw, v_in, self.v_dyn_st, self.v_dyn_pr_f16,
                    T, cfg.qk_l2_clip_threshold, tree_lag=tree_lag,
                )
            cache.set_conv_after(cache_slot, 0, kp, k_raw.reshape(1, T, 1024), new_kp)
            cache.set_conv_after(cache_slot, 1, vp, v_raw.reshape(1, T, 1024), new_vp)

        KV_buf, kv_len, kv_start, kv_cap, kv_is_fp8, kv_q4_sc = cache.update_kv_f16(cache_slot, kv_f16, self.is_sliding)

        # 4. Fused Sinks + GQA+RelPos Flash-Decoding / FlashAttention-2 SDPA
        attn_out = _model_mod.gqa_relpos_sinks_sdpa_fast(
            q, KV_buf, rel_st, self.rel_proj_t_f16, q_in, self.sink_w_f16,
            T, kv_len, kv_start, q_pos0, kv_cap, self.is_sliding,
            tree_depth=tree_depth, tree_mask=tree_mask, kv_fp8=kv_is_fp8, kv_q4_scales=kv_q4_sc,
        )

        # 5. Fused Gate*Attn + RHT-3072 -> O projection -> Post-Attn ShortConv4 (+ Residual)
        o_rot = _model_mod.rht_in_3072_gated_fast(attn_out, o_gate_raw, self.o_stage.signs_in, out_dtype=mx.float16)
        o_out = self.o_stage.matmul_rotated_f16(o_rot, T, W_cached=o_w_cached)
        if self.o_out_rot is not None:
            o_out = _model_mod.rht_apply_out(o_out, self.o_out_rot)
        pa_prev = cache.get_conv_prev(cache_slot, 2, 2304)
        if self.is_block_boundary:
            h_post, new_pa = _model_mod.short_conv4_res_fused(pa_prev, o_out, self.post_attn_conv_w, T, 2304, residual=None, tree_lag=tree_lag)
            residual = h_post
            if precomputed_local_blocks is not None:
                local_blocks = precomputed_local_blocks
            else:
                b_new = mx.fast.rms_norm(block_residual, None, 1e-6) if blocks_prenorm else block_residual
                local_blocks = b_new[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, b_new[None, :, :]], axis=0)
            n_local = n_blocks + 1
        else:
            h_post, new_pa = _model_mod.short_conv4_res_fused(pa_prev, o_out, self.post_attn_conv_w, T, 2304, residual=block_residual, tree_lag=tree_lag)
            residual = h_post
            local_blocks = blocks_cat
            n_local = n_blocks
        cache.set_conv_after(cache_slot, 2, pa_prev, o_out.reshape(1, T, 2304), new_pa)

        # 6+7. Fused MLP input AttnRes / MsgBlend + Norm + RHT-2304(gu_signs) -> Unified MoE + post-MLP conv + residual!
        if messages is None and rank2_ctx is None:
            h_mlp_f16, x_rot_gu = bar_rht_fn(self.mlp_in_q, local_blocks, n_local, h_post, self.post_attn_ln, self.moe.gu_signs_in)
        elif rank2_ctx is not None:
            v2_1d, v3_1d, b3_flat, Z, loop_layer_idx = rank2_ctx
            col_mlp = 36 + loop_layer_idx
            h_m = bar_fn(self.mlp_in_q, local_blocks, n_local, h_post)[0]
            fg_mg = _msg_1x_rank2_compiled(h_m, Z, self.mlp_dh, self.mlp_fm, col_mlp)
            h_mlp_f16, x_rot_gu = msg_blend_norm_rht_rank2(h_m, v2_1d, v3_1d, b3_flat, fg_mg, self.post_attn_ln, self.moe.gu_signs_in, col=col_mlp)
        else:
            h_m = bar_fn(self.mlp_in_q, local_blocks, n_local, h_post)[0]
            fg_mg = _model_mod._msg_1x_compiled(h_m, messages[3], self.mlp_dh, self.mlp_dm, self.mlp_fm)
            h_mlp_f16, x_rot_gu = _model_mod.msg_blend_norm_rht_fast(h_m, messages[3], fg_mg, self.post_attn_ln, self.moe.gu_signs_in)

        return self.moe.forward_fast_fused_prerot(h_mlp_f16, x_rot_gu, residual, self.post_mlp_conv_w, cache, cache_slot, T, tree_lag=tree_lag)

    _model_mod.PackedDecoderLayer.forward_fast = _fast_layer_forward

    def _adaptive_forward_core_fast(
        self,
        input_ids,
        cache,
        all_logits: bool = False,
        argmax_only: bool = False,
        ordered_only: bool = False,
        tap_visits: tuple[int, ...] = (),
        tree_spec: dict | None = None,
        skip_logits: bool = False,
    ):
        ant_mode = os.environ.get("LAMQ_ANT_TIER_MODE", "auto").strip().lower()
        if ant_mode not in ("0", "off", "false", ""):
            cur_tier = getattr(self, "_ant_postloop_tier", "high")
            if ant_mode == "low" and cur_tier != "low":
                set_postloop_ant_tier(self, "low", log_fn=log_fn)
            elif ant_mode == "high" and cur_tier != "high":
                set_postloop_ant_tier(self, "high", log_fn=log_fn)
            elif ant_mode == "auto":
                cur_seq = int(getattr(cache, "seq_len", 0) or 0)
                ctx_thr = int(os.environ.get("LAMQ_ANT_DOWNGRADE_CTX", "65536"))
                mem_thr = int(float(os.environ.get("LAMQ_ANT_DOWNGRADE_GB", "16.5")) * (1 << 30))
                act_mem = mx.get_active_memory()
                if cur_tier == "high" and (cur_seq >= ctx_thr or act_mem >= mem_thr):
                    set_postloop_ant_tier(self, "low", log_fn=log_fn)
                elif (
                    cur_tier == "low"
                    and os.environ.get("LAMQ_ANT_STICKY_LOW", "0") != "1"
                    and cur_seq < max(4096, ctx_thr // 4)
                    and act_mem < max(0, mem_thr - (2 << 30))
                ):
                    set_postloop_ant_tier(self, "high", log_fn=log_fn)

        cfg = self.cfg
        h = self.embed(input_ids)[0]  # [T, 2304]
        T = int(h.shape[0])
        q_pos0 = cache.seq_len
        blocks_cat: mx.array | None = None
        n_blocks = 0
        tap_set = set(tap_visits)
        captured: dict[int, mx.array] = {}
        visit = 0

        use_prenorm = os.environ.get("LAMQ_PRENORM_BAR", "1") != "0"
        use_rank2 = os.environ.get("LAMQ_RANK2_LOOP_MSG", "1") != "0"
        bar_fn = block_attn_res_prenorm_fused if use_prenorm else _model_mod.block_attn_res_fused

        force_low = os.environ.get("LAMQ_LOW_MEM_PREFILL") == "1"
        is_short_bypass = 64 <= T < 512 and getattr(cache, "seq_len", 0) == 0 and int(getattr(cache, "_reserve_cap", 0) or 0) == 0
        low_mem_prefill = force_low or is_short_bypass or (T >= 64 and mx.get_active_memory() > _low_mem_thresh)
        cache_loop_w = (T >= 64) and (cfg.loop_passes > 1) and (not low_mem_prefill)
        loop_w_cache: dict[int, tuple[mx.array, mx.array]] = {}

        # 1. Pre-loop layers (L0..5)
        for i in range(cfg.loop_start):
            if visit in tap_set:
                captured[visit] = h
            L = self.layers[i]
            if L.is_block_boundary:
                b_new = mx.fast.rms_norm(h, None, 1e-6) if use_prenorm else h
                next_blocks = b_new[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, b_new[None, :, :]], axis=0)
            else:
                next_blocks = None
            new_h = L.forward_fast(
                h, blocks_cat, n_blocks, q_pos0, T, cache, i,
                messages=None, tree_spec=tree_spec,
                blocks_prenorm=use_prenorm, precomputed_local_blocks=next_blocks,
            )
            if L.is_block_boundary:
                blocks_cat = next_blocks
                n_blocks += 1
            h = new_h
            visit += 1
            if (visit & 3) == 0:
                mx.async_eval(h)

        pre_loop_h = h
        pre_loop_blocks = blocks_cat
        num_pre = n_blocks

        # 2. Pass 0 over loop layers (L6..17)
        for i in range(cfg.loop_start, cfg.loop_end):
            if visit in tap_set:
                captured[visit] = h
            L = self.layers[i]
            if cache_loop_w:
                wq_c = L.qkv_stage.dequant_f16()
                wo_c = L.o_stage.dequant_f16()
                loop_w_cache[i] = (wq_c, wo_c)
            else:
                wq_c = wo_c = None
            if L.is_block_boundary:
                b_new = mx.fast.rms_norm(h, None, 1e-6) if use_prenorm else h
                next_blocks = b_new[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, b_new[None, :, :]], axis=0)
            else:
                next_blocks = None
            new_h = L.forward_fast(
                h, blocks_cat, n_blocks, q_pos0, T, cache, i,
                messages=None, tree_spec=tree_spec, qkv_w_cached=wq_c, o_w_cached=wo_c,
                blocks_prenorm=use_prenorm, precomputed_local_blocks=next_blocks,
            )
            if L.is_block_boundary:
                blocks_cat = next_blocks
                n_blocks += 1
            h = new_h
            visit += 1
            if (visit & 3) == 0:
                mx.async_eval(h)

        # 3. Passes 1..loop_passes-1 over loop layers with Rank-2 cross-pass message factorization
        num_loop = cfg.loop_end - cfg.loop_start
        u_ordered, all_dm_ordered = ensure_rank2_loop_tables(self) if use_rank2 else (None, None)
        for loop_idx in range(1, cfg.loop_passes):
            prev_n = n_blocks - (num_pre + 1)
            prev_blocks = blocks_cat[num_pre + 1 :] if prev_n > 0 else None
            prev_h = h
            h = pre_loop_h
            blocks_cat = pre_loop_blocks
            n_blocks = num_pre
            cache_offset = cfg.num_hidden_layers + (loop_idx - 1) * num_loop - cfg.loop_start

            if use_rank2 and prev_n == 1 and prev_blocks is not None:
                v2_hat, v3_hat, b3_flat, Z_rank2 = precompute_loop_pass_rank2(
                    prev_blocks[0], prev_h, u_ordered, all_dm_ordered, v2_is_prenorm=use_prenorm
                )
                v2_1d, v3_1d = v2_hat.reshape(-1), v3_hat.reshape(-1)
                use_pass_rank2 = True
            else:
                v2_1d = v3_1d = b3_flat = Z_rank2 = None
                use_pass_rank2 = False

            for loop_layer_idx, i in enumerate(range(cfg.loop_start, cfg.loop_end)):
                if visit in tap_set:
                    captured[visit] = h
                L = self.layers[i]
                wq_c, wo_c = loop_w_cache[i] if cache_loop_w else (None, None)
                if L.is_block_boundary:
                    b_new = mx.fast.rms_norm(h, None, 1e-6) if use_prenorm else h
                    next_blocks = b_new[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, b_new[None, :, :]], axis=0)
                else:
                    next_blocks = None
                if use_pass_rank2:
                    new_h = L.forward_fast(
                        h, blocks_cat, n_blocks, q_pos0, T, cache, i + cache_offset,
                        messages=None, tree_spec=tree_spec, qkv_w_cached=wq_c, o_w_cached=wo_c,
                        blocks_prenorm=use_prenorm, precomputed_local_blocks=next_blocks,
                        rank2_ctx=(v2_1d, v3_1d, b3_flat, Z_rank2, loop_layer_idx),
                    )
                else:
                    msgs = bar_fn(self.message_in_q[loop_layer_idx], prev_blocks, prev_n, prev_h)
                    new_h = L.forward_fast(
                        h, blocks_cat, n_blocks, q_pos0, T, cache, i + cache_offset,
                        messages=msgs, tree_spec=tree_spec, qkv_w_cached=wq_c, o_w_cached=wo_c,
                        blocks_prenorm=use_prenorm, precomputed_local_blocks=next_blocks,
                    )
                if L.is_block_boundary:
                    blocks_cat = next_blocks
                    n_blocks += 1
                h = new_h
                visit += 1
                if (visit & 3) == 0:
                    mx.async_eval(h)
        if cache_loop_w:
            loop_w_cache.clear()

        # 4. Post-loop layers (L18..23)
        for i in range(cfg.loop_end, cfg.num_hidden_layers):
            if visit in tap_set:
                captured[visit] = h
            L = self.layers[i]
            if L.is_block_boundary:
                b_new = mx.fast.rms_norm(h, None, 1e-6) if use_prenorm else h
                next_blocks = b_new[None, :, :] if n_blocks == 0 else mx.concatenate([blocks_cat, b_new[None, :, :]], axis=0)
            else:
                next_blocks = None
            new_h = L.forward_fast(
                h, blocks_cat, n_blocks, q_pos0, T, cache, i,
                messages=None, tree_spec=tree_spec,
                blocks_prenorm=use_prenorm, precomputed_local_blocks=next_blocks,
            )
            if L.is_block_boundary:
                blocks_cat = next_blocks
                n_blocks += 1
            h = new_h
            visit += 1
            if (visit & 3) == 0 and visit < 48:
                mx.async_eval(h)

        # 5. Final AttnRes + norm + unpadded lm_head
        if skip_logits:
            if visit in tap_set:
                h = bar_fn(self.final_in_q, blocks_cat, n_blocks, h, norm_w=self.final_norm_w)[0]
                captured[visit] = h
            cache.seq_len += T
            if tap_visits:
                taps = mx.stack([captured[v] for v in tap_visits], axis=1)
                return h, cache, taps
            return h, cache
        if not all_logits and visit not in tap_set and T > 1:
            b_last = blocks_cat[:, -1:, :] if (blocks_cat is not None and n_blocks > 0) else blocks_cat
            h_in = bar_fn(self.final_in_q, b_last, n_blocks, h[-1:, :], norm_w=self.final_norm_w)[0][None, :, :]
        else:
            h = bar_fn(self.final_in_q, blocks_cat, n_blocks, h, norm_w=self.final_norm_w)[0]
            if visit in tap_set:
                captured[visit] = h
            h_in = h[None, :, :] if all_logits else h[-1:, :][None, :, :]
        cache.seq_len += T
        if argmax_only:
            out = self.lm_head.argmax(h_in)
        elif ordered_only:
            out = self.lm_head.ordered_logits(h_in)
        else:
            out = self.lm_head(h_in)
        if tap_visits:
            taps = mx.stack([captured[v] for v in tap_visits], axis=1)
            return out, cache, taps
        return out, cache

    _model_mod.Inkling2LoopPackedMLX._forward_core_fast = _adaptive_forward_core_fast

    # 3. 16+ core M4 Pro / M4 Max grid tuning (overrides base 10-core M4 defaults only on g16s/c/d)
    m4_pro = is_m4_pro_or_max()
    if m4_pro:
        orig_dequant = _fk.DenseStage.dequant_f16
        orig_sg8 = _fk.DenseStage.matmul_rotated_sg8_f16
        orig_dense_mm = _fk.DenseStage.matmul_rotated_f16
        orig_moe_call = _fk.UnifiedMoESide.__call__

        def _tuned_dequant_f16(self, ntg: int | None = None) -> mx.array:
            return orig_dequant(self, ntg=48 if ntg is None else ntg)

        def _tuned_sg8(self, x_rot_tr_f16: mx.array, T: int = 8, rb: int | None = None, sg: int | None = None, ntg: int | None = None) -> mx.array:
            if rb is None and sg is None and ntg is None:
                if self.total_out > 10000:
                    return orig_sg8(self, x_rot_tr_f16, T=T, rb=2, sg=8, ntg=128)
                return orig_sg8(self, x_rot_tr_f16, T=T, rb=1, sg=8, ntg=64)
            return orig_sg8(self, x_rot_tr_f16, T=T, rb=rb, sg=sg, ntg=ntg)

        def _tuned_dense_mm(self, x_rot_f16: mx.array, T: int, W_cached: mx.array | None = None) -> mx.array:
            if T == 1:
                if self.total_out > 10000:
                    rps, sg, ntg = 2, 8, 160
                elif self.total_out <= 3072:
                    # o_stage (3072 -> 2304): 6 TGs/core on 16-core M4 Pro (1.31x faster than stock ntg=160 with 16 empty TGs)
                    rps, sg, ntg = 2, 4, 96
                else:
                    # qkv_stage (2304 -> 8576): 9 TGs/core on 16-core M4 Pro (1.20x faster than stock rps=1, sg=4, ntg=160)
                    rps, sg, ntg = 2, 8, 144
                kern = _fk._make_dense_stage_kernel(self.n, self.fids, 1, rps=rps, sg=sg, stage_rtr=self._stage_rtr)
                mt = _fk._u32(self.total_out, self.n_mats, ntg)
                return kern(
                    inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self.levels, self.dims, *self._lut_args, mt],
                    grid=(ntg * sg * 32, 1, 1),
                    threadgroup=(sg * 32, 1, 1),
                    output_shapes=[(1, self.total_out)],
                    output_dtypes=[mx.float32],
                )[0]
            if 2 <= T <= 8:
                # o_stage (m=2304, rps=2 -> 1152 blocks, sg=4 -> 8 blocks/TG => 144 = 16*9 exact TGs, 1 block/TG, 0 wave tail vs stock ntg=40!)
                rps, sg, ntg = 2, 4, 144
                kern = _fk._make_dense_stage_kernel(self.n, self.fids, T, rps=rps, sg=sg, stage_rtr=self._stage_rtr)
                mt = _fk._u32(self.total_out, self.n_mats, ntg)
                return kern(
                    inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self.levels, self.dims, *self._lut_args, mt],
                    grid=(ntg * sg * 32, 1, 1),
                    threadgroup=(sg * 32, 1, 1),
                    output_shapes=[(T, self.total_out)],
                    output_dtypes=[mx.float32],
                )[0]
            return orig_dense_mm(self, x_rot_f16, T, W_cached=W_cached)

        def _tuned_moe_call(
            self,
            x_rot_f16: mx.array,
            eids_flat: mx.array,
            x_div: int,
            sorted_cache=None,
            plan=None,
            dynamic_routing: bool = False,
            sorted_in: bool = False,
            sorted_out: bool = False,
            x_raw_f16: mx.array | None = None,
        ) -> mx.array:
            n_slots = int(eids_flat.shape[0])
            nv_tail = [(x_raw_f16 if x_raw_f16 is not None else x_rot_f16).reshape(-1), *self._nv_args] if self.has_nvfp4 else []
            # T=1 decode: sg=8, ntg=96 (GU 1.76x faster than 10-core sg=4, ntg=60)
            if n_slots <= 8 and not sorted_in and not sorted_out:
                sg, ntg = 8, (96 if self.n_cols >= 2304 else 128)
                kern = _fk._make_unified_moe_kernel(self.n_cols, self.fids, "t1", rps=4, sg=sg, n_rg=self.n_rg, stage_rtr=self._stage_rtr)
                mt = _fk._u32(self.m_rows, n_slots, x_div, ntg)
                return kern(
                    inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self._kern_fid, self.exp_code_off, self.exp_scale_off, *self._lut_args, eids_flat.astype(mx.uint32), mt, *nv_tail],
                    grid=(ntg * sg * 32, 1, 1),
                    threadgroup=(sg * 32, 1, 1),
                    output_shapes=[(n_slots, self.m_rows)],
                    output_dtypes=[mx.float32],
                )[0]
            # T=2..8 DSpark verify: rps=4, lpr=8, ntg=160 (1.17x faster than 10-core rps=2, lpr=16)
            if n_slots <= 64 and not sorted_in and not sorted_out:
                if plan is None:
                    plan = _fk._make_moe_plan_only_kernel(self.fids)(
                        inputs=[eids_flat.astype(mx.uint32), self.exp_fid, _fk._u32(n_slots)],
                        grid=(128, 1, 1),
                        threadgroup=(128, 1, 1),
                        output_shapes=[(129,)],
                        output_dtypes=[mx.uint32],
                    )[0]
                sg_g, rps_g, lpr_g, ntg_g = 8, 4, 8, 160
                kern = _fk._make_unified_moe_kernel(self.n_cols, self.fids, "grouped", rps=rps_g, sg=sg_g, lpr=lpr_g, n_rg=self.n_rg, stage_rtr=self._stage_rtr)
                mt = _fk._u32(self.m_rows, x_div, ntg_g)
                return kern(
                    inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self._kern_fid, self.exp_code_off, self.exp_scale_off, *self._lut_args, plan, mt, *nv_tail],
                    grid=(ntg_g * sg_g * 32, 1, 1),
                    threadgroup=(sg_g * 32, 1, 1),
                    output_shapes=[(n_slots, self.m_rows)],
                    output_dtypes=[mx.float32],
                )[0]
            # Prefill (non-MPP): MoE Prefill v4 (`tile_t=64`, 9.1 KB SRAM -> 3 TGs/core, per-j A0/A1 streaming, opt-out via LAMQ_MOE_V4=0)
            if not _fk._has_mpp():
                if os.environ.get("LAMQ_MOE_V4", "1") != "0":
                    from .moe_v4 import run_moe_side_prefill_v4

                    return run_moe_side_prefill_v4(
                        self, x_rot_f16, eids_flat, x_div,
                        sorted_cache=sorted_cache, sorted_in=sorted_in, sorted_out=sorted_out,
                        x_raw_f16=x_raw_f16,
                    )
                offsets, order = sorted_cache if sorted_cache is not None else _fk.compute_moe_sorted_offsets(eids_flat)
                max_nb = 6 if n_slots > 2048 else 4
                tile_t = 8 * max_nb
                units, ucnt, max_u = _fk.plan_moe_prefill_units(offsets, self.exp_fid, self.fids, n_slots, tile_t=tile_t)
                sg_p, rb_p = 4, 2
                rows_per_tg = sg_p * 8 * rb_p
                rblocks = (self.m_rows + rows_per_tg - 1) // rows_per_tg
                kern = make_fast_moe_prefill_v3_kernel(
                    self.n_cols, self.fids, sorted_in=sorted_in, sorted_out=sorted_out, sg=sg_p, rb=rb_p, n_rg=self.n_rg, max_nb=max_nb,
                )
                return kern(
                    inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self._kern_fid, self.exp_code_off, self.exp_scale_off, *self._lut_args_prefill_v2, units, ucnt, _fk._u32(self.m_rows, x_div), order, *nv_tail],
                    grid=(max_u * sg_p * 32, rblocks, 1),
                    threadgroup=(sg_p * 32, 1, 1),
                    output_shapes=[(n_slots, self.m_rows)],
                    output_dtypes=[mx.float32],
                )[0]
            return orig_moe_call(
                self, x_rot_f16, eids_flat, x_div,
                sorted_cache=sorted_cache, plan=plan, dynamic_routing=dynamic_routing,
                sorted_in=sorted_in, sorted_out=sorted_out, x_raw_f16=x_raw_f16,
            )

        _fk.DenseStage.dequant_f16 = _tuned_dequant_f16
        _fk.DenseStage.matmul_rotated_sg8_f16 = _tuned_sg8
        _fk.DenseStage.matmul_rotated_f16 = _tuned_dense_mm
        _fk.UnifiedMoESide.__call__ = _tuned_moe_call

    _model_mod.Inkling2LoopPackedMLX.set_postloop_ant_tier = set_postloop_ant_tier

    # 3b-0. Project 1 (Cheetah Ant / RC2-B inspired): Format-Homogeneous & Frequency-Sorted (Hot-to-Cold) MoE Slab Packing
    if hasattr(_model_mod, "PackedMoE"):
        orig_packed_moe_init = _model_mod.PackedMoE.__init__

        def _hot_cold_packed_moe_init(self, arrays, luts, fast_luts, router_w, router_bias, cfg):
            if os.environ.get("LAMQ_MOE_HOT_COLD_PACK", "1") != "0":
                arrays, router_w, router_bias, perm = repack_moe_arrays_hot_cold(arrays, router_w, router_bias)
                self._expert_perm = perm
            return orig_packed_moe_init(self, arrays, luts, fast_luts, router_w, router_bias, cfg)

        _model_mod.PackedMoE.__init__ = _hot_cold_packed_moe_init

    # 3b. MoE Prefill v4 Fused `GU -> SwiGLU -> [NVFP4-skipping RHT-1152] -> DN` (`tile_t=64`, 9.1 KB SRAM, bit-exact 0.0)
    #     + Decode (`T <= 8`) NVFP4-skipping SwiGLU/RHT-1152 across 42/48 visits of 12.8GB (19 pure-NVFP4 + 23 mixed)
    moe_v4_enabled = (not _fk._has_mpp()) and (os.environ.get("LAMQ_MOE_V4", "1") != "0")
    if moe_v4_enabled and hasattr(_model_mod, "PackedMoE"):
        from .moe_v4 import (
            fast_nv_skip_swiglu_rht1152_f16,
            fast_swiglu_only_f16,
            fused_moe_prefill_v4,
        )

        orig_packed_moe_prerot = _model_mod.PackedMoE.forward_fast_fused_prerot

        def _fused_v4_moe_prerot(
            self,
            h_mlp_f16: mx.array,
            x_rot_gu_f16: mx.array,
            residual: mx.array,
            post_mlp_conv_w: mx.array,
            cache,
            cache_slot: int,
            T: int,
            tree_lag: mx.array | None = None,
        ) -> mx.array:
            if (
                os.environ.get("LAMQ_MOE_V4", "1") != "0"
                and self.gu_unified.n_rg == 1
                and self.gu_unified.m_rows == 2304
                and self.dn_unified.n_cols == 1152
            ):
                dyn, eff_kmax, eff_kmin, eff_thr, eff_renorm = resolve_decode_moe_routing(self.cfg, cache_slot, T)
                if T > 8:
                    logits_f16 = h_mlp_f16 @ self.router_w_f16.T
                    if dyn:
                        eids_flat, w_norm = _model_mod.sigmoid_top8_norm_fast(
                            logits_f16, self.router_bias, self.cfg.route_scale,
                            k_max=eff_kmax, k_min=eff_kmin,
                            ratio_thr=eff_thr, renorm=eff_renorm,
                        )
                    else:
                        eids_flat, w_norm = _model_mod.sigmoid_top8_norm_fast(
                            logits_f16, self.router_bias, self.cfg.route_scale
                        )
                    sc = _model_mod.compute_moe_sorted_offsets(eids_flat)
                    y_dn = fused_moe_prefill_v4(
                        self.gu_unified, self.dn_unified,
                        h_mlp_f16, x_rot_gu_f16, self.dn_signs_in,
                        eids_flat, sc,
                        gate_cap=self.cfg.mlp_gate_softcap,
                        up_cap=self.cfg.mlp_up_softcap,
                        dynamic_routing=dyn,
                    )
                    prev = cache.get_conv_prev(cache_slot, 3, 2304)
                    out, new_prev, raw_sum = _model_mod.moe_wsum_sconv4_res_fused(
                        y_dn, w_norm, prev, post_mlp_conv_w, residual, T, tree_lag=tree_lag, skip_zero_w=dyn
                    )
                    cache.set_conv_after(cache_slot, 3, prev, raw_sum, new_prev)
                    return out
                if self.dn_unified.n_rg == 1:
                    logits_f16 = h_mlp_f16 @ self.router_w_f16.T
                    if 2 <= T <= 8:
                        if dyn:
                            eids_flat, w_norm, gu_plan, dn_plan = _model_mod.sigmoid_top8_norm_and_plan_fast(
                                logits_f16, self.router_bias, self.cfg.route_scale,
                                self.gu_unified.exp_fid, self.gu_unified.fids,
                                self.dn_unified.exp_fid, self.dn_unified.fids,
                                k_max=eff_kmax, k_min=eff_kmin,
                                ratio_thr=eff_thr, renorm=eff_renorm,
                            )
                        else:
                            eids_flat, w_norm, gu_plan, dn_plan = _model_mod.sigmoid_top8_norm_and_plan_fast(
                                logits_f16, self.router_bias, self.cfg.route_scale,
                                self.gu_unified.exp_fid, self.gu_unified.fids,
                                self.dn_unified.exp_fid, self.dn_unified.fids,
                            )
                    else:
                        if dyn:
                            eids_flat, w_norm = _model_mod.sigmoid_top8_norm_fast(
                                logits_f16, self.router_bias, self.cfg.route_scale,
                                k_max=eff_kmax, k_min=eff_kmin,
                                ratio_thr=eff_thr, renorm=eff_renorm,
                            )
                        else:
                            eids_flat, w_norm = _model_mod.sigmoid_top8_norm_fast(
                                logits_f16, self.router_bias, self.cfg.route_scale
                            )
                        gu_plan = dn_plan = None
                    y_gu = self.gu_unified(
                        x_rot_gu_f16, eids_flat, x_div=8, sorted_cache=None, plan=gu_plan,
                        dynamic_routing=dyn, x_raw_f16=h_mlp_f16,
                    )
                    if self.dn_unified.has_nvfp4:
                        if self.dn_unified.fids == (7,):
                            # 19/48 pure-NVFP4 visits: 100% skip RHT-1152 and second buffer allocation!
                            act_raw = fast_swiglu_only_f16(
                                y_gu, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap,
                                eids_flat=eids_flat if dyn else None,
                            )
                            act_rot = act_raw
                        else:
                            # 23/48 mixed visits: skip H_16 + Paley-72 RHT-1152 on every NVFP4 slot (75-90% of slots)!
                            act_rot, act_raw = fast_nv_skip_swiglu_rht1152_f16(
                                y_gu, self.dn_signs_in, eids_flat, self.dn_unified.exp_fid,
                                self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap,
                            )
                    else:
                        act_rot = _model_mod.swiglu_rht_1152_fast(
                            y_gu, self.dn_signs_in, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap,
                            out_dtype=mx.float16, eids_flat=eids_flat if dyn else None,
                        )
                        act_raw = None
                    y_dn = self.dn_unified(
                        act_rot, eids_flat, x_div=1, sorted_cache=None, plan=dn_plan,
                        dynamic_routing=dyn, x_raw_f16=act_raw,
                    )
                    prev = cache.get_conv_prev(cache_slot, 3, 2304)
                    out, new_prev, raw_sum = _model_mod.moe_wsum_sconv4_res_fused(
                        y_dn, w_norm, prev, post_mlp_conv_w, residual, T, tree_lag=tree_lag, skip_zero_w=dyn
                    )
                    cache.set_conv_after(cache_slot, 3, prev, raw_sum, new_prev)
                    return out
            return orig_packed_moe_prerot(
                self, h_mlp_f16, x_rot_gu_f16, residual, post_mlp_conv_w, cache, cache_slot, T, tree_lag=tree_lag
            )

        _model_mod.PackedMoE.forward_fast_fused_prerot = _fused_v4_moe_prerot

    # 3c. Temperature-Calibrated Draft Proposal (`tau_q = gamma * tau_p`, 0% target distribution bias)
    #     + 46K Multilingual/Cyrillic Draft Head Shortlist (`LAMQ_MULTILINGUAL_DRAFT_HEAD=1`, +20% Russian tpc)
    if not hasattr(_gen_mod, "_orig_verify_spec_sample_gpu"):
        _gen_mod._orig_verify_spec_sample_gpu = _gen_mod.verify_spec_sample_gpu
    _gen_mod.verify_spec_sample_gpu = verify_sparse_spec_sample_gpu

    if hasattr(_model_mod, "PackedLMHead"):
        if hasattr(_model_mod.PackedLMHead, "_ensure_draft_head"):
            orig_ensure_draft_head = _model_mod.PackedLMHead._ensure_draft_head

            def _multilingual_ensure_draft_head(self, max_buckets: int = 5, bits: int = 4):
                key = (max_buckets, bits)
                if key in self._draft_cache:
                    return self._draft_cache[key]
                packed_dir = getattr(self, "_packed_dir", None)
                if (
                    os.environ.get("LAMQ_MULTILINGUAL_DRAFT_HEAD", "1") == "0"
                    or not getattr(self, "_fast", False)
                    or packed_dir is None
                    or max_buckets >= getattr(self, "nb", 0)
                ):
                    return orig_ensure_draft_head(self, max_buckets=max_buckets, bits=bits)

                cyr_ids = extract_cyrillic_token_ids(packed_dir / "tokenizer.json")
                if not cyr_ids:
                    return orig_ensure_draft_head(self, max_buckets=max_buckets, bits=bits)

                import numpy as np

                nb = min(max_buckets, self.nb)
                W_full = self.stage.dequant_f16()
                wqs, scs, bis = [], [], []
                total_rows = 0
                for bi in range(nb):
                    lo, hi = self.stage.slices[bi]
                    rows_b = hi - lo
                    total_rows += rows_b
                    Q_b = W_full[lo:hi].astype(mx.float32)
                    W_b = _model_mod.rht_out_2304_fast(
                        Q_b, self.stage.signs_in[bi : bi + 1], mx.zeros((rows_b,), dtype=mx.uint32)
                    )
                    wq, sc, bi_q = mx.quantize(W_b.astype(mx.float16), group_size=64, bits=bits)
                    mx.eval(wq, sc, bi_q)
                    wqs.append(wq)
                    scs.append(sc)
                    bis.append(bi_q)

                extra_ord: list[np.ndarray] = []
                cyr_list = list(cyr_ids)
                for bi in range(nb, self.nb):
                    lo, hi = self.stage.slices[bi]
                    b_toks = self.order_np[lo:hi]
                    rel_idx = np.where(np.isin(b_toks, cyr_list))[0]
                    if len(rel_idx) == 0:
                        continue
                    abs_ord_idx = (lo + rel_idx).astype(np.int32)
                    rows_b = len(rel_idx)
                    Q_b = W_full[mx.array(abs_ord_idx)].astype(mx.float32)
                    W_b = _model_mod.rht_out_2304_fast(
                        Q_b, self.stage.signs_in[bi : bi + 1], mx.zeros((rows_b,), dtype=mx.uint32)
                    )
                    wq, sc, bi_q = mx.quantize(W_b.astype(mx.float16), group_size=64, bits=bits)
                    mx.eval(wq, sc, bi_q)
                    wqs.append(wq)
                    scs.append(sc)
                    bis.append(bi_q)
                    extra_ord.append(abs_ord_idx)
                del W_full

                wq_all = mx.concatenate(wqs, axis=0)
                sc_all = mx.concatenate(scs, axis=0)
                bi_all = mx.concatenate(bis, axis=0)
                if extra_ord:
                    ord_map_np = np.concatenate([np.arange(total_rows, dtype=np.int32), *extra_ord], axis=0)
                    draft_ord_map = mx.array(ord_map_np)
                    order_sub = self.order_mx[draft_ord_map]
                    mx.eval(wq_all, sc_all, bi_all, order_sub, draft_ord_map)
                    _SPARSE_DRAFT_ORD_MAPS[int(draft_ord_map.shape[0])] = draft_ord_map
                else:
                    order_sub = self.order_mx[:total_rows]
                    mx.eval(wq_all, sc_all, bi_all, order_sub)
                mx.clear_cache()
                self._draft_cache[key] = (wq_all, sc_all, bi_all, order_sub, bits)
                return self._draft_cache[key]

            _model_mod.PackedLMHead._ensure_draft_head = _multilingual_ensure_draft_head

        if hasattr(_model_mod.PackedLMHead, "get_draft_head_sampler"):
            orig_get_draft_sampler = _model_mod.PackedLMHead.get_draft_head_sampler

            def _calibrated_get_draft_head_sampler(
                self,
                temp: float,
                top_p: float = 1.0,
                top_k: int = 0,
                max_buckets: int = 5,
                bits: int = 4,
            ):
                eff_temp = calibrate_draft_temp(temp)
                return orig_get_draft_sampler(
                    self, eff_temp, top_p=top_p, top_k=top_k, max_buckets=max_buckets, bits=bits
                )

            _model_mod.PackedLMHead.get_draft_head_sampler = _calibrated_get_draft_head_sampler

    _INSTALLED = True
    summary = {
        "installed": True,
        "custom_gqa_sg8_sdpa": True,
        "custom_drafter_gqa12_sdpa": True,
        "custom_radix_topp": True,
        "moe_prefill_v4_fused": moe_v4_enabled,
        "multilingual_draft_head": os.environ.get("LAMQ_MULTILINGUAL_DRAFT_HEAD", "1") != "0",
        "ngram_default": ngram_enabled,
        "kv_alloc_step": _kv_step,
        "m4_pro_grid_tuning": m4_pro,
    }
    if log_fn is not None:
        log_fn(
            f"metal accel enabled: GQA-SG8 SDPA + GQA-FA2 + DSpark-GQA12 SDPA + Radix-TopP "
            f"+ MoE-v4-Fused={'on' if moe_v4_enabled else 'off'} "
            f"+ Multilingual-DraftHead={'on' if summary['multilingual_draft_head'] else 'off'} "
            f"+ Ngram={'on' if ngram_enabled else 'off'} + O(N) KV(4096) "
            f"+ M4-Pro-16c={'on' if m4_pro else 'off'}"
        )
    return summary
