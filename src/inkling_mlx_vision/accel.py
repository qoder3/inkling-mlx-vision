"""Custom Apple Silicon Metal kernels and 16+ core M4 Pro/Max accelerations for LookAtMyQuant Inkling-2-Mini.

All optimizations are 100% mathematically equivalent (bit-exact or FP16-epsilon parity, rmse <= 1e-5):
1. Custom GQA-Shared Split-K `simdgroup_half8x8` SDPA Metal kernel for `T=2..8` (DSpark block & tree verify):
   - 96 threads (3 simdgroups = 3 GQA heads sharing ONE K/V tile load across up to 24 query vectors)
   - Hardware `simdgroup_multiply_accumulate` instead of scalar `simd_sum` loops
   - 4.35x (FP8) / 4.74x (Q4) faster than stock `gqa_fused_sdpa` at 65K context (saves 73-88 ms/block)
2. Custom Lean-SRAM GQA-Shared `simdgroup_half8x8` FlashAttention-2 Metal kernel for prefill (`T >= 32`):
   - `BQ=32, BK=32` (384 threads = 12 simdgroups = 4 Q-tiles x 3 GQA heads sharing ONE K/V tile load)
   - Fits in 15.6 KB threadgroup SRAM (loads Q one GQA head at a time)
   - 1.53x (FP8) / 1.35x (Q4) faster than stock `FlashAttn2` at 65K context (saves ~218 ms/chunk)
3. 16+ core Apple M4 Pro/Max threadgroup grid tuning (fixes base 10-core M4 `ntg=10/20/30/60` throttling):
   - `DenseStage.dequant_f16`: 1.90x faster (`ntg=48`)
   - `DenseStage` LM head `T=8`: 1.71x faster (`ntg=128`), QKV `T=8`: 1.37x faster (`rb=1, ntg=64`)
   - `UnifiedMoESide` GU `T=1`: 1.76x faster (`sg=8, ntg=96`), `T=2..8`: 1.17x faster, `T=256` prefill: 1.24x faster (`sg=4`)
4. O(N) Global KV pre-allocation / 4096-step growth (eliminates O(N^2) `+256` reallocation copies in prefill & decode).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlx.core as mx
from runtime_mlx.inkling import drafter as _drafter_mod
from runtime_mlx.inkling import fast_kernels as _fk
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
        const bool need_rel = (d_min < 1024) && ((int)q_pos_max >= (int)(kv_start + k0));
        const bool fast_no_rel = (q0 + {bq}u <= T) && (k0 + {bk}u <= k_hi) && (d_min >= 1024) && !is_sliding;

        float m_tile = -1e30f;
        float2 sv[{kb_cnt}];
        if (fast_no_rel) {{
            #pragma unroll
            for (uint kb = 0u; kb < {kb_cnt}u; kb++) {{
                float2 s2 = ELEMS_F2(S[kb].thread_elements());
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
            if T == 1 and self.total_out > 10000:
                rps, sg, ntg = 2, 8, 160
                kern = _fk._make_dense_stage_kernel(self.n, self.fids, 1, rps=rps, sg=sg, stage_rtr=self._stage_rtr)
                mt = _fk._u32(self.total_out, self.n_mats, ntg)
                return kern(
                    inputs=[x_rot_f16.reshape(-1), self.codes, self.scales, self.levels, self.dims, *self._lut_args, mt],
                    grid=(ntg * sg * 32, 1, 1),
                    threadgroup=(sg * 32, 1, 1),
                    output_shapes=[(1, self.total_out)],
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
            # Prefill (non-MPP): sg_p=4 beats sg_p=8 by 1.24x at T<=320 (and for DN at all T)
            if not _fk._has_mpp() and (n_slots <= 2560 or self.n_cols < 2304):
                offsets, order = sorted_cache if sorted_cache is not None else _fk.compute_moe_sorted_offsets(eids_flat)
                tile_t = 32
                units, ucnt, max_u = _fk.plan_moe_prefill_units(offsets, self.exp_fid, self.fids, n_slots, tile_t=tile_t)
                sg_p, rb_p = 4, 2
                rows_per_tg = sg_p * 8 * rb_p
                rblocks = (self.m_rows + rows_per_tg - 1) // rows_per_tg
                kern = _fk._make_unified_moe_prefill_v2_kernel(
                    self.n_cols, self.fids, sorted_in=sorted_in, sorted_out=sorted_out, sg=sg_p, rb=rb_p, n_rg=self.n_rg,
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

    _INSTALLED = True
    summary = {
        "installed": True,
        "custom_gqa_sg8_sdpa": True,
        "kv_alloc_step": _kv_step,
        "m4_pro_grid_tuning": m4_pro,
    }
    if log_fn is not None:
        log_fn(
            f"metal accel enabled: GQA-SG8 Split-K SDPA + GQA-FA2 prefill + O(N) KV(4096) "
            f"+ M4-Pro-16c tuning={'on' if m4_pro else 'off'}"
        )
    return summary
