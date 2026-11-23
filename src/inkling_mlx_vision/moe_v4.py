"""Custom Metal Kernel 8 & 9: MoE Prefill v4 (`tile_t=64`, 9.1 KB SRAM, Fused `GU -> SwiGLU` + NVFP4-skipping `RHT-1152`).

100% bit-exact (`max_diff == 0.0`) improvements over MoE Prefill v3 (`tile_t=48`, unfused):
  1. Exact-sized LUT union in threadgroup SRAM: aliases `tg_ht4` (4 KB), `tg_rtr` (8 KB), and NVFP4
     256-entry `half2` byte-pair LUT `tg_e2m1_p` + `tg_e4m3` (1.5 KB) into the minimum needed size
     (4 KB on all 24 layers of 12.8GB and 22/24 layers of 8.9GB; 1.5 KB on pure-NVFP4 layers).
  2. Per-`j` `A0, A1` register streaming (`nb_val >= 5`) + non-unrolled epilogue: holds only 2 `Am`
     registers + 1 `Xm` register while loading `Xm` once per `(b, j)` across both `RB=2` row blocks,
     unlocking `tile_t = 64` (`nb_val = 1..8`) with zero register spilling inside 9.1 KB single-buffered
     threadgroup SRAM (3 threadgroups/core occupancy on Apple M4 Pro!).
  3. Fused `GU -> Softcapped SwiGLU` (`sorted_out=True`): pairs `gate` (`r0`) and `up` (`r0 + 1152`)
     in the `RB=2` accumulators and computes softcapped SwiGLU in registers — completely eliminating
     the `[n_slots, 2304] float32` `y_gu` tensor (saves 37.7–75.5 MB peak RAM/visit & up to 6.3 GB
     DRAM traffic/chunk) and keeping `GU -> RHT -> DN` 100% coalesced (`sorted_out=True` -> `sorted_in=True`).
  4. NVFP4-skipping `RHT-1152-only` (`make_rht1152_from_raw_kernel`):
     - On pure-NVFP4 `dn` layers (`dn.fids == (7,)`, 19/48 visits of 12.8GB): `GU -> SwiGLU` writes
       `float16` `act_raw` directly and skips `RHT-1152` completely (`0.00 ms`, `max_diff == 0.0`).
     - On mixed `ht/rtr + nvfp4` `dn` layers (23/48 visits of 12.8GB): skips `H_16 + Paley-72` on all
       NVFP4 slots while emitting `act_raw_f16` in the same pass (`max_diff == 0.0`).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
from runtime_mlx.inkling import fast_kernels as _fk


@lru_cache(maxsize=96)
def make_fast_moe_prefill_v4_kernel(
    n: int,
    fids: tuple[int, ...],
    sorted_in: bool = False,
    sorted_out: bool = False,
    sg: int = 4,
    rb: int = 2,
    n_rg: int = 1,
    max_nb: int = 8,
    fused_swiglu: bool = False,
    out_half: bool = False,
    single_buf: bool | None = None,
):
    """High-occupancy Apple Silicon Metal MoE Prefill v4 kernel (`tile_t=64`, per-`j` `A0/A1`, optional fused SwiGLU)."""
    if fused_swiglu:
        assert rb == 2 and n_rg == 1, "fused_swiglu requires rb=2 and n_rg=1"
    steps = n // 32
    tok_stride = sg * 4
    max_tok_cap = 8 * max_nb
    nb_tuple = tuple(range(1, max_nb + 1))
    has_ht = any(f in fids for f in (0, 1, 2))
    has_rtr = any(f in fids for f in (3, 4, 5))
    has_nv = 7 in fids

    # Exact-sized union LUT (bytes): rtr=8192, ht=4096, nv(pair-LUT)=1536, rtr4-only=8
    lut_bytes = 8192 if has_rtr else (4096 if has_ht else (1536 if has_nv else 8))
    tg_x_dbl = 2 * max_tok_cap * 36 * 2
    idx_bytes = 2 * max_tok_cap * 4
    # Single-buffer `tg_x` for `max_nb >= 7` (9.1 KB -> 3 TGs/core on M4 Pro!) or whenever double-buf > 16 KB.
    use_single = (max_nb >= 7 or (tg_x_dbl + lut_bytes + idx_bytes > 16384)) if single_buf is None else bool(single_buf)
    out_ty = "half" if (fused_swiglu and out_half) else "float"

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

    def _emit_decode_am_unstreamed(fid: int, nb_val: int):
        k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
        wpr_c = (n * k) // 32
        def _mac(j: int, expr: str):
            ha(f"                ELEMS_H2(Am.thread_elements()) = {expr};")
            ha("                #pragma unroll")
            ha(f"                for (uint b = 0; b < {nb_val}u; b++) simdgroup_multiply_accumulate(Acc[i][b], Am, Xm[b][{j}], Acc[i][b]);")
        if fid in (0, 1):
            if fid == 1:
                ha(f"                const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
            else:
                ha(f"                const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u; const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            ha(f"                const half4 v0 = tg_ht4[((extract_bits(lo, {4 * k}u, 12u)) * {_fk.HASH_MULT_HEX}) >> 23u];")
            ha(f"                const half4 v1 = tg_ht4[((extract_bits(lo, 0u, 12u)) * {_fk.HASH_MULT_HEX}) >> 23u];")
            for j, expr in [(0, "v0.xy"), (1, "v0.zw"), (2, "v1.xy"), (3, "v1.zw")]:
                _mac(j, expr)
        elif fid == 2:
            ha(f"                const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            ha("                const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                _mac(j, f"tg_ht2[(extract_bits(lo, {rb_b}u, 12u) * {_fk.HASH_MULT_HEX}) >> 22u]")
        elif fid == 3:
            ha("                const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> off) | (w_prev[i] << (32u - off))); w_prev[i] = w0;")
            for j, sh in [(0, 6), (1, 4), (2, 2), (3, 0)]:
                _mac(j, _fk._rtr_value(3, 2, f"extract_bits(lo, {sh}u, 13u)"))
        elif fid == 4:
            ha(f"                const uint w0 = pw[0]; const uint lo = (off == 0u) ? w0 : ((w0 >> 16u) | (((wi + 1u < {wpr_c}u) ? pw[1] : 0u) << 16u));")
            for j, rb_b in [(0, 12), (1, 8), (2, 4), (3, 0)]:
                _mac(j, _fk._rtr_value(4, 2, f"extract_bits(lo, {rb_b}u, 14u)"))
        elif fid == 5:
            ha(f"                const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u, w2 = (wi + 2u < {wpr_c}u) ? pw[2] : 0u;")
            ha("                const uint lo = (w0 >> off) | (off == 0u ? 0u : (w1 << (32u - off)));")
            ha("                const uint hi = (w1 >> off) | (off == 0u ? 0u : (w2 << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                ex = f"extract_bits(lo, {rb_b}u, 15u)" if rb_b + 15 <= 32 else f"insert_bits(extract_bits(lo, {rb_b}u, {32 - rb_b}u), hi, {32 - rb_b}u, {15 - (32 - rb_b)}u)"
                _mac(j, _fk._rtr_value(5, 2, ex))
        elif fid == 6:
            ha(f"                const uint w0 = pw[0], w1 = (wi + 1u < {wpr_c}u) ? pw[1] : 0u;")
            for j, (b0, b1) in [(0, (28, 24)), (1, (20, 16)), (2, (12, 8)), (3, (4, 0))]:
                ex0 = f"extract_bits(w0, {b0}u, 12u)" if b0 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b0}u, {32 - b0}u), w1, {32 - b0}u, {12 - (32 - b0)}u)"
                ex1 = f"extract_bits(w0, {b1}u, 12u)" if b1 + 12 <= 32 else f"insert_bits(extract_bits(w0, {b1}u, {32 - b1}u), w1, {32 - b1}u, {12 - (32 - b1)}u)"
                _mac(j, f"half2(lut_rtr4[{ex0}], lut_rtr4[{ex1}])")

    def _emit_jstream_step(fid: int, nb_val: int, cur_off_expr: str):
        """Per-j `Am[RB]` streaming: only `RB` (=2) temp A regs + 1 Xm reg, 1x SRAM load of Xm per (b, j)."""
        def _emit_j_loop(j: int, exprs: list[str]):
            ha("            {")
            ha("                simdgroup_half8x8 A[RB];")
            for i_idx, ex in enumerate(exprs):
                ha(f"                ELEMS_H2(A[{i_idx}].thread_elements()) = {ex};")
            ha(f"                for (uint b = 0; b < {nb_val}u; b++) {{")
            ha(f"                    simdgroup_half8x8 Xm; simdgroup_load(Xm, tg_x + {cur_off_expr} + (8u * b) * XSTR + {8 * j}u, XSTR, ulong2(0, 0), true);")
            for i_idx in range(rb):
                ha(f"                    simdgroup_multiply_accumulate(Acc[{i_idx}][b], A[{i_idx}], Xm, Acc[{i_idx}][b]);")
            ha("                }")
            ha("            }")

        if fid == 7:
            for i_idx in range(rb):
                ha(f"            const uint w0_{i_idx} = p_c[{i_idx}][4u * s + q]; const half bs_{i_idx} = tg_e4m3[p_s[{i_idx}][2u * s + (q >> 1)]];")
            for j in range(4):
                _emit_j_loop(j, [f"tg_e2m1_p[extract_bits(w0_{i_idx}, {8 * j}u, 8u)] * bs_{i_idx}" for i_idx in range(rb)])
            return

        k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
        wpr_c = (n * k) // 32
        ha(f"            const uint wi = w_first - s * {k}u;")
        if fid in (0, 1):
            for i_idx in range(rb):
                ha(f"            const device uint* pw_{i_idx} = p[{i_idx}] + wi;")
                if fid == 1:
                    ha(f"            const uint w0_{i_idx} = pw_{i_idx}[0]; const uint lo_{i_idx} = (off == 0u) ? w0_{i_idx} : ((w0_{i_idx} >> 16u) | (((wi + 1u < {wpr_c}u) ? pw_{i_idx}[1] : 0u) << 16u));")
                else:
                    ha(f"            const uint w0_{i_idx} = pw_{i_idx}[0], w1_{i_idx} = (wi + 1u < {wpr_c}u) ? pw_{i_idx}[1] : 0u; const uint lo_{i_idx} = (w0_{i_idx} >> off) | (off == 0u ? 0u : (w1_{i_idx} << (32u - off)));")
            for half_idx, bit_sh in [(0, 4 * k), (1, 0)]:
                for i_idx in range(rb):
                    ha(f"            const half4 hv{half_idx}_{i_idx} = tg_ht4[((extract_bits(lo_{i_idx}, {bit_sh}u, 12u)) * {_fk.HASH_MULT_HEX}) >> 23u];")
                _emit_j_loop(half_idx * 2, [f"hv{half_idx}_{i_idx}.xy" for i_idx in range(rb)])
                _emit_j_loop(half_idx * 2 + 1, [f"hv{half_idx}_{i_idx}.zw" for i_idx in range(rb)])
        elif fid == 2:
            for i_idx in range(rb):
                ha(f"            const device uint* pw_{i_idx} = p[{i_idx}] + wi; const uint w0_{i_idx} = pw_{i_idx}[0], w1_{i_idx} = (wi + 1u < {wpr_c}u) ? pw_{i_idx}[1] : 0u;")
                ha(f"            const uint lo_{i_idx} = (w0_{i_idx} >> off) | (off == 0u ? 0u : (w1_{i_idx} << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                _emit_j_loop(j, [f"tg_ht2[(extract_bits(lo_{i_idx}, {rb_b}u, 12u) * {_fk.HASH_MULT_HEX}) >> 22u]" for i_idx in range(rb)])
        elif fid == 3:
            for i_idx in range(rb):
                ha(f"            const uint w0_{i_idx} = p[{i_idx}][wi]; const uint lo_{i_idx} = (off == 0u) ? w0_{i_idx} : ((w0_{i_idx} >> off) | (w_prev[{i_idx}] << (32u - off))); w_prev[{i_idx}] = w0_{i_idx};")
            for j, sh in [(0, 6), (1, 4), (2, 2), (3, 0)]:
                _emit_j_loop(j, [_fk._rtr_value(3, 2, f"extract_bits(lo_{i_idx}, {sh}u, 13u)") for i_idx in range(rb)])
        elif fid == 4:
            for i_idx in range(rb):
                ha(f"            const device uint* pw_{i_idx} = p[{i_idx}] + wi; const uint w0_{i_idx} = pw_{i_idx}[0];")
                ha(f"            const uint lo_{i_idx} = (off == 0u) ? w0_{i_idx} : ((w0_{i_idx} >> 16u) | (((wi + 1u < {wpr_c}u) ? pw_{i_idx}[1] : 0u) << 16u));")
            for j, rb_b in [(0, 12), (1, 8), (2, 4), (3, 0)]:
                _emit_j_loop(j, [_fk._rtr_value(4, 2, f"extract_bits(lo_{i_idx}, {rb_b}u, 14u)") for i_idx in range(rb)])
        elif fid == 5:
            for i_idx in range(rb):
                ha(f"            const device uint* pw_{i_idx} = p[{i_idx}] + wi; const uint w0_{i_idx} = pw_{i_idx}[0], w1_{i_idx} = (wi + 1u < {wpr_c}u) ? pw_{i_idx}[1] : 0u, w2_{i_idx} = (wi + 2u < {wpr_c}u) ? pw_{i_idx}[2] : 0u;")
                ha(f"            const uint lo_{i_idx} = (w0_{i_idx} >> off) | (off == 0u ? 0u : (w1_{i_idx} << (32u - off)));")
                ha(f"            const uint hi_{i_idx} = (w1_{i_idx} >> off) | (off == 0u ? 0u : (w2_{i_idx} << (32u - off)));")
            for j, rb_b in [(0, 18), (1, 12), (2, 6), (3, 0)]:
                exprs = []
                for i_idx in range(rb):
                    ex = f"extract_bits(lo_{i_idx}, {rb_b}u, 15u)" if rb_b + 15 <= 32 else f"insert_bits(extract_bits(lo_{i_idx}, {rb_b}u, {32 - rb_b}u), hi_{i_idx}, {32 - rb_b}u, {15 - (32 - rb_b)}u)"
                    exprs.append(_fk._rtr_value(5, 2, ex))
                _emit_j_loop(j, exprs)
        elif fid == 6:
            for i_idx in range(rb):
                ha(f"            const device uint* pw_{i_idx} = p[{i_idx}] + wi; const uint w0_{i_idx} = pw_{i_idx}[0], w1_{i_idx} = (wi + 1u < {wpr_c}u) ? pw_{i_idx}[1] : 0u;")
            for j, (b0, b1) in [(0, (28, 24)), (1, (20, 16)), (2, (12, 8)), (3, (4, 0))]:
                exprs = []
                for i_idx in range(rb):
                    ex0 = f"extract_bits(w0_{i_idx}, {b0}u, 12u)" if b0 + 12 <= 32 else f"insert_bits(extract_bits(w0_{i_idx}, {b0}u, {32 - b0}u), w1_{i_idx}, {32 - b0}u, {12 - (32 - b0)}u)"
                    ex1 = f"extract_bits(w0_{i_idx}, {b1}u, 12u)" if b1 + 12 <= 32 else f"insert_bits(extract_bits(w0_{i_idx}, {b1}u, {32 - b1}u), w1_{i_idx}, {32 - b1}u, {12 - (32 - b1)}u)"
                    exprs.append(f"half2(lut_rtr4[{ex0}], lut_rtr4[{ex1}])")
                _emit_j_loop(j, exprs)

    caps_arg = ", float4 caps4" if fused_swiglu else ""
    row_expr = "(r0 + half_m * i)" if fused_swiglu else "min(r0 + 8u * i, m_rows - 1u)"

    for fid in fids:
        is_nv = fid == 7
        if not is_nv:
            k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
            wpr_c = (n * k) // 32
        else:
            wpr_nv, spr_nv = n // 8, n // 16
        for nb_val in nb_tuple:
            max_t = 8 * nb_val
            use_stream = nb_val >= 5
            if is_nv:
                ha(
                    f"inline void umoe_v4_f7_nb{nb_val}("
                    "const device half* xbase, const device uint* ecodes, const device uchar* env_scales, float eouter, "
                    "threadgroup half* tg_x, const threadgroup half2* tg_e2m1_p, const threadgroup half* tg_e4m3, "
                    f"const threadgroup uint* tg_xrow, const threadgroup uint* tg_out_slot, device {out_ty}* y, "
                    f"uint rbase, uint m_rows, uint y_stride, uint cnt, uint tid, uint lane{caps_arg}) {{"
                )
            else:
                ha(
                    f"inline void umoe_v4_f{fid}_nb{nb_val}("
                    "const device half* xbase, const device uint* ecodes, const device half* escales, "
                    "threadgroup half* tg_x, const threadgroup half4* tg_ht4, const threadgroup half2* tg_ht2, "
                    f"{_fk._RTR_ARGS}const device half* lut_rtr4, "
                    f"const threadgroup uint* tg_xrow, const threadgroup uint* tg_out_slot, device {out_ty}* y, "
                    f"uint rbase, uint m_rows, uint y_stride, uint cnt, uint tid, uint lane{caps_arg}) {{"
                )
            ha("    const uint tok_i = tid >> 3, vec_k = tid & 7u, vk_q = vec_k >> 1, vk_h = vec_k & 1u;")
            ha("    const uint qid = lane / 4u, fm = (qid & 4u) + ((lane / 2u) % 4u), fn = (qid & 2u) * 2u + (lane % 2u) * 2u;")
            ha("    const uint q = fn / 2u, r0 = rbase + fm;")
            if fused_swiglu:
                ha("    const uint half_m = y_stride;")
            if is_nv:
                ha("    const device uint* p_c[RB]; const device uchar* p_s[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) {{ const uint rr = min({row_expr}, m_rows - 1u); p_c[i] = ecodes + (ulong)rr * {wpr_nv}u; p_s[i] = env_scales + (ulong)rr * {spr_nv}u; }}")
            else:
                ha("    const device uint* p[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) p[i] = ecodes + (ulong)min({row_expr}, m_rows - 1u) * {wpr_c}u;")
                ha(f"    const uint first = ({n}u - 8u - 8u * q) * {k}u, w_first = first >> 5, off = first & 31u;")
            ha(f"    simdgroup_float8x8 Acc[RB][{nb_val}];")
            ha(f"    for (uint i = 0; i < RB; i++) for (uint b = 0; b < {nb_val}u; b++) Acc[i][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
            if fid == 3:
                ha("    uint w_prev[RB];")
                ha(f"    for (uint i = 0; i < RB; i++) w_prev[i] = (w_first + 1u < {wpr_c}u) ? p[i][w_first + 1u] : 0u;")
            if not use_single:
                _emit_x_load(max_t, "xbase", "0u", "0u")
                ha("    threadgroup_barrier(mem_flags::mem_threadgroup);")
            ha(f"    for (uint s = 0; s < {steps}u; s++) {{")
            if use_single:
                _emit_x_load(max_t, "xbase", "32u * s", "0u")
                ha("        threadgroup_barrier(mem_flags::mem_threadgroup);")
                cur_off_expr = "0u"
            else:
                ha("        const uint cur_off = (s & 1u) * PING_STRIDE, nxt_off = ((s + 1u) & 1u) * PING_STRIDE;")
                ha(f"        if (s + 1u < {steps}u) {{")
                _emit_x_load(max_t, "xbase", "32u * (s + 1u)", "nxt_off")
                ha("        }")
                cur_off_expr = "cur_off"
            row_lim = "half_m" if fused_swiglu else "m_rows"
            ha(f"        if (rbase < {row_lim}) {{")
            if not use_stream:
                ha(f"            simdgroup_half8x8 Xm[{nb_val}][4];")
                ha("            #pragma unroll")
                ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                ha("                #pragma unroll")
                ha(f"                for (uint j = 0; j < 4u; j++) simdgroup_load(Xm[b][j], tg_x + {cur_off_expr} + (8u * b) * XSTR + 8u * j, XSTR, ulong2(0, 0), true);")
                ha("            }")
                if is_nv:
                    ha("            #pragma unroll")
                    ha("            for (uint i = 0; i < RB; i++) {")
                    ha("                const uint w0 = p_c[i][4u * s + q]; const half bsc = tg_e4m3[p_s[i][2u * s + (q >> 1)]]; simdgroup_half8x8 Am;")
                    for j in range(4):
                        ha(f"                ELEMS_H2(Am.thread_elements()) = tg_e2m1_p[extract_bits(w0, {8 * j}u, 8u)] * bsc;")
                        ha("                #pragma unroll")
                        ha(f"                for (uint b = 0; b < {nb_val}u; b++) simdgroup_multiply_accumulate(Acc[i][b], Am, Xm[b][{j}], Acc[i][b]);")
                    ha("            }")
                else:
                    ha(f"            const uint wi = w_first - s * {k}u;")
                    ha("            #pragma unroll")
                    ha("            for (uint i = 0; i < RB; i++) {")
                    ha("                const device uint* pw = p[i] + wi; simdgroup_half8x8 Am;")
                    _emit_decode_am_unstreamed(fid, nb_val)
                    ha("            }")
            else:
                _emit_jstream_step(fid, nb_val, cur_off_expr)
            ha("        }")
            ha("        threadgroup_barrier(mem_flags::mem_threadgroup);")
            ha("    }")
            # Epilogue (deliberately NOT `#pragma unroll` to prevent Apple GPU register-file cliff!)
            if fused_swiglu:
                ha("    if (r0 < half_m) {")
                if is_nv:
                    ha("        const float sc_g = eouter, sc_u = eouter;")
                else:
                    ha("        const float sc_g = (float)escales[r0], sc_u = (float)escales[r0 + half_m];")
                ha(f"        for (uint b = 0; b < {nb_val}u; b++) {{")
                ha("            const uint ta = 8u * b + fn, tb = ta + 1u;")
                ha("            const float2 g = ELEMS_F2(Acc[0][b].thread_elements()) * sc_g;")
                ha("            const float2 u = ELEMS_F2(Acc[1][b].thread_elements()) * sc_u;")
                ha("            const float2 ra = (metal::precise::tanh(g * caps4.y) * caps4.x) * (1.0f / (1.0f + metal::precise::exp(-g))) * (metal::precise::tanh(u * caps4.w) * caps4.z);")
                ha(f"            if (ta < cnt) y[(ulong)tg_out_slot[ta] * half_m + r0] = ({out_ty})ra.x;")
                ha(f"            if (tb < cnt) y[(ulong)tg_out_slot[tb] * half_m + r0] = ({out_ty})ra.y;")
                ha("        }")
                ha("    }")
            else:
                ha("    if (rbase < m_rows) {")
                ha("        for (uint i = 0; i < RB; i++) {")
                ha("            const uint row = r0 + 8u * i; if (row >= m_rows) continue;")
                if not is_nv:
                    ha("            const float sc = (float)escales[row];")
                ha(f"            for (uint b = 0; b < {nb_val}u; b++) {{")
                ha("                const uint ta = 8u * b + fn, tb = ta + 1u;")
                sc_e = "eouter" if is_nv else "sc"
                ha(f"                const float2 d = ELEMS_F2(Acc[i][b].thread_elements()) * {sc_e};")
                ha("                if (ta < cnt) y[(ulong)tg_out_slot[ta] * y_stride + row] = d.x;")
                ha("                if (tb < cnt) y[(ulong)tg_out_slot[tb] * y_stride + row] = d.y;")
                ha("            }")
                ha("        }")
                ha("    }")
            ha("}")

    lut_h4 = (lut_bytes + 7) // 8
    tg_x_elems = (1 if use_single else 2) * max_tok_cap
    L = [
        "const uint u = threadgroup_position_in_grid.x, rb_tg = threadgroup_position_in_grid.y;",
        "if (u >= unit_count[0]) return;",
        "const uint simd = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup, tid = simd * 32u + lane;",
        "const uint m_rows = meta[0], x_div = meta[1];",
        "const uint eid = units[u * 3u + 0u], base = units[u * 3u + 1u], cnt = units[u * 3u + 2u];",
    ]
    if fused_swiglu:
        L += [
            "const uint fid = exp_fid[eid], rbase = rb_tg * (SG * 8u) + simd * 8u, m_grp = m_rows, y_str = m_rows >> 1;",
            f"device {out_ty}* y_ptr = y;",
        ]
    elif n_rg > 1:
        L += [
            f"const uint rg_sz = m_rows / {n_rg}u, r_tg = rb_tg * (SG * 8u * RB), rg = r_tg / rg_sz;",
            f"const uint rbase = (r_tg - rg * rg_sz) + simd * (8u * RB), m_grp = rg_sz, y_str = m_rows, idx_rg = eid * {n_rg}u + rg;",
            "const uint fid = exp_fid[idx_rg];",
            "device float* y_ptr = y + rg * rg_sz;",
        ]
    else:
        L += [
            "const uint fid = exp_fid[eid], rbase = rb_tg * (SG * 8u * RB) + simd * (8u * RB), m_grp = m_rows, y_str = m_rows;",
            "device float* y_ptr = y;",
        ]
    L += [
        "const device half4* t_ht1 = (const device half4*)lut_ht1; const device half4* t_ht2 = (const device half4*)lut_ht2;",
        f"threadgroup half tg_x[{tg_x_elems} * XSTR];",
        f"threadgroup half4 tg_lut_u[{lut_h4}];",
        "threadgroup half4* tg_ht4 = tg_lut_u; const threadgroup half2* tg_ht2 = (const threadgroup half2*)tg_lut_u;",
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
            "threadgroup half2* tg_e2m1_p = (threadgroup half2*)tg_lut_u; threadgroup half* tg_e4m3 = (threadgroup half*)(tg_e2m1_p + 256u);",
            f"if (fid == 7u) {{ for (uint idx = tid; idx < 256u; idx += {sg * 32}u) {{ tg_e2m1_p[idx] = half2(lut_e2m1[idx & 15u], lut_e2m1[idx >> 4]); tg_e4m3[idx] = lut_e4m3[idx]; }} }}",
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
    idx_expr = "idx_rg" if (n_rg > 1 and not fused_swiglu) else "eid"
    caps_pass = ", float4(caps[0], caps[1], caps[2], caps[3])" if fused_swiglu else ""
    L.append(f"const device uint* ec = codes + exp_code_off[{idx_expr}]; const device half* es = scales + exp_scale_off[{idx_expr}];")
    L.append("switch (fid) {")
    for fid in fids:
        L.append(f"    case {fid}u:")
        if fid == 7:
            L.append("        { const device uchar* env_s = nv_scales + exp_nv_scale_off[eid]; const float eout = nv_outer[eid];")
            for nb_v in nb_tuple[:-1]:
                kw = "if" if nb_v == 1 else "else if"
                L.append(f"          {kw} (cnt <= {8 * nb_v}u) umoe_v4_f7_nb{nb_v}({xbase_expr}, ec, env_s, eout, tg_x, tg_e2m1_p, tg_e4m3, tg_xrow, tg_out_slot, y_ptr, rbase, m_grp, y_str, cnt, tid, lane{caps_pass});")
            L.append(f"          else umoe_v4_f7_nb{max_nb}({xbase_expr}, ec, env_s, eout, tg_x, tg_e2m1_p, tg_e4m3, tg_xrow, tg_out_slot, y_ptr, rbase, m_grp, y_str, cnt, tid, lane{caps_pass}); }}")
        else:
            for nb_v in nb_tuple[:-1]:
                kw = "if" if nb_v == 1 else "else if"
                L.append(f"        {kw} (cnt <= {8 * nb_v}u) umoe_v4_f{fid}_nb{nb_v}({xbase_expr}, ec, es, tg_x, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, tg_xrow, tg_out_slot, y_ptr, rbase, m_grp, y_str, cnt, tid, lane{caps_pass});")
            L.append(f"        else umoe_v4_f{fid}_nb{max_nb}({xbase_expr}, ec, es, tg_x, tg_ht4, tg_ht2, tv_rtr1, tv_rtr2, tv_rtr3, lut_rtr4, tg_xrow, tg_out_slot, y_ptr, rbase, m_grp, y_str, cnt, tid, lane{caps_pass});")
        L.append("        break;")
    L.append("}")
    in_names = ["x", "codes", "scales", "exp_fid", "exp_code_off", "exp_scale_off", "lut_ht1", "lut_ht2", "lut_ht3", "lut_rtr1", "lut_rtr2", "lut_rtr3", "lut_rtr4", "units", "unit_count", "meta", "slot_order"]
    if has_nv:
        in_names += ["x_raw", "nv_scales", "exp_nv_scale_off", "nv_outer", "lut_e2m1", "lut_e4m3"]
    if fused_swiglu:
        in_names += ["caps"]
    tag = "_".join(str(f) for f in fids)
    return mx.fast.metal_kernel(
        name=f"ink_umoe_v4_n{n}_f{tag}_si{int(sorted_in)}_so{int(sorted_out)}_rb{rb}_s{sg}_rg{n_rg}_mnb{max_nb}_fu{int(fused_swiglu)}_oh{int(out_half)}_sb{int(use_single)}",
        input_names=in_names,
        output_names=["y"],
        header="\n".join(H),
        source="\n".join(L),
    )


@lru_cache(maxsize=16)
def make_rht1152_from_raw_kernel(
    in_half: bool = False,
    out_half: bool = True,
    also_raw_f16: bool = False,
    skip_mode: int = 0,
):
    """Paley-72 RHT-1152 directly from pre-computed SwiGLU `raw_in` (0 tanh/exp, 2-4x less input traffic).

    `skip_mode`:
      - 0: transform all slots.
      - 1: skip sentinel slots `slot >= offsets[128]` (for dynamic routing).
      - 2: skip sentinel slots AND skip `H_16 + Paley-72` for NVFP4 `dn` experts (`dn_fid[e] == 7u`),
           still writing `raw_out[slot] = half4(rp4[tid])` when `also_raw_f16=True`.
    """
    it4 = "half4" if in_half else "float4"
    ot4 = "half4" if out_half else "float4"
    conv = "half4(tz4[tid] * INV_S)" if out_half else "(tz4[tid] * INV_S)"
    extra_in: list[str] = []
    if skip_mode == 1:
        extra_in = ["offsets"]
        skip_code = "    if (slot >= offsets[128]) return;\n"
    elif skip_mode == 2:
        extra_in = ["offsets", "slot_order", "eids", "dn_fid"]
        nv_raw_store = (
            "((device half4*)(raw_out + (ulong)slot * 1152u))[tid] = half4(((const device "
            + it4
            + "*)(raw_in + (ulong)slot * 1152u))[tid]); "
            if also_raw_f16
            else ""
        )
        skip_code = (
            "    if (slot >= offsets[128]) return;\n"
            "    const uint os = slot_order[slot];\n"
            "    const uint e = eids[os];\n"
            "    if (e >= 128u) return;\n"
            f"    if (dn_fid[e] == 7u) {{ {nv_raw_store}return; }}\n"
        )
    else:
        skip_code = ""
    raw_store = "\n    ((device half4*)(raw_out + (ulong)slot * 1152u))[tid] = half4(raw_v);" if also_raw_f16 else ""
    src = f"""
    const uint slot = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = simd * 32u + lane;
{skip_code}    const uint hw = lane >> 4, hl = lane & 15u;
    const device {it4}* rp4 = (const device {it4}*)(raw_in + (ulong)slot * 1152u);
    const device float4* sp4 = (const device float4*)signs;
    device {ot4}* op4 = (device {ot4}*)(out + (ulong)slot * 1152u);

    threadgroup float tg_z[16 * 73];
    const float4 raw_v = float4(rp4[tid]);{raw_store}
    const float4 act = raw_v * sp4[tid];
    const uint a = tid / 18u, c = (tid - a * 18u) * 4u;
    threadgroup float* dst = tg_z + a * 73u + c;
    dst[0] = act.x; dst[1] = act.y; dst[2] = act.z; dst[3] = act.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c0 = simd * 8u + hw * 4u;
    const bool h1 = hl & 1u, h2 = hl & 2u, h4 = hl & 4u, h8 = hl & 8u;
    #pragma unroll
    for (uint j = 0; j < 4u; j++) {{
        const uint cc = c0 + j;
        float v = tg_z[hl * 73u + cc];
        float o;
        o = simd_shuffle_xor(v, 1u); v = h1 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u); v = h2 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u); v = h4 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u); v = h8 ? (o - v) : (v + o);
        tg_z[hl * 73u + cc] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma unroll
    for (uint kb = 0; kb < 9u; kb++) {{
        simdgroup_float8x8 Bm, Zm0, Zm1;
        simdgroup_load(Bm, B + (simd * 8u) * 72u + kb * 8u, 72u, ulong2(0, 0), true);
        simdgroup_load(Zm0, tg_z + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_load(Zm1, tg_z + 8u * 73u + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(Acc0, Zm0, Bm, Acc0);
        simdgroup_multiply_accumulate(Acc1, Zm1, Bm, Acc1);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(Acc0, tg_z + simd * 8u, 72u);
    simdgroup_store(Acc1, tg_z + 8u * 72u + simd * 8u, 72u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr float INV_S = 0.02946278254943948f;
    const threadgroup float4* tz4 = (const threadgroup float4*)tg_z;
    op4[tid] = {conv};
    """
    out_names = ["out", "raw_out"] if also_raw_f16 else ["out"]
    return mx.fast.metal_kernel(
        name=f"ink_rht1152_raw_ih{int(in_half)}_oh{int(out_half)}_ar{int(also_raw_f16)}_sm{skip_mode}",
        input_names=["raw_in", "signs", "B"] + extra_in,
        output_names=out_names,
        header="#include <metal_simdgroup_matrix>\n",
        source=src,
    )


def run_moe_side_prefill_v4(
    side,
    x_rot_f16: mx.array,
    eids_flat: mx.array,
    x_div: int,
    *,
    sorted_cache: tuple[mx.array, mx.array] | None = None,
    sorted_in: bool = False,
    sorted_out: bool = False,
    x_raw_f16: mx.array | None = None,
    fused_swiglu: bool = False,
    out_half: bool = False,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> mx.array:
    """Execute one `UnifiedMoESide` prefill pass with MoE Prefill v4 (`tile_t=64`, 9.1 KB SRAM, optional fused SwiGLU)."""
    n_slots = int(eids_flat.shape[0])
    offsets, order = sorted_cache if sorted_cache is not None else _fk.compute_moe_sorted_offsets(eids_flat)
    # Optimal M4 Pro schedule (verified on real 48-visit 12.8GB Inkling-2-Mini):
    #   T <= 64  (n_slots <= 512):  tile_t=32 (dbl, 7.5 KB)
    #   T <= 512 (n_slots <= 4096): tile_t=48 (dbl, 11.1 KB -> +19.5% over v3 at T=512, -90 MB peak)
    #   T > 512  (n_slots > 4096):  tile_t=64 (sng,  9.1 KB -> 3 TGs/core, 446 tok/s at T=1024, -80 MB peak)
    max_nb = 8 if n_slots > 4096 else (6 if n_slots > 512 else 4)
    tile_t = 8 * max_nb
    units, ucnt, max_u = _fk.plan_moe_prefill_units(offsets, side.exp_fid, side.fids, n_slots, tile_t=tile_t)
    sg_p, rb_p = 4, 2
    m_out = (side.m_rows // 2) if fused_swiglu else side.m_rows
    rows_per_tg = sg_p * 8 * (1 if fused_swiglu else rb_p)
    rblocks = (m_out + rows_per_tg - 1) // rows_per_tg
    kern = make_fast_moe_prefill_v4_kernel(
        side.n_cols, side.fids,
        sorted_in=sorted_in, sorted_out=sorted_out,
        sg=sg_p, rb=rb_p, n_rg=side.n_rg, max_nb=max_nb,
        fused_swiglu=fused_swiglu, out_half=out_half,
    )
    nv_tail = [(x_raw_f16 if x_raw_f16 is not None else x_rot_f16).reshape(-1), *side._nv_args] if side.has_nvfp4 else []
    caps_tail = [_fk._f32_arr(gate_cap, 1.0 / gate_cap, up_cap, 1.0 / up_cap)] if fused_swiglu else []
    out_dtype = mx.float16 if (fused_swiglu and out_half) else mx.float32
    return kern(
        inputs=[
            x_rot_f16.reshape(-1), side.codes, side.scales, side._kern_fid,
            side.exp_code_off, side.exp_scale_off, *side._lut_args_prefill_v2,
            units, ucnt, _fk._u32(side.m_rows, x_div), order, *nv_tail, *caps_tail,
        ],
        grid=(max_u * sg_p * 32, rblocks, 1),
        threadgroup=(sg_p * 32, 1, 1),
        output_shapes=[(n_slots, m_out)],
        output_dtypes=[out_dtype],
    )[0]


def fused_moe_prefill_v4(
    gu_side,
    dn_side,
    h_mlp_f16: mx.array,
    x_rot_gu_f16: mx.array,
    dn_signs_in: mx.array,
    eids_flat: mx.array,
    sorted_cache: tuple[mx.array, mx.array],
    *,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
    dynamic_routing: bool = False,
) -> mx.array:
    """Bit-exact (`max_diff == 0.0`) fused `GU -> SwiGLU -> [NVFP4-skipping RHT-1152] -> DN` prefill pipeline."""
    n_slots = int(eids_flat.shape[0])
    offsets, order = sorted_cache
    pure_nv_dn = dn_side.fids == (7,)
    need_dn_raw = dn_side.has_nvfp4

    # 1. Fused GU -> Softcapped SwiGLU (sorted_out=True, zero y_gu[n_slots, 2304] f32 allocation!)
    raw_sorted = run_moe_side_prefill_v4(
        gu_side, x_rot_gu_f16, eids_flat, x_div=8,
        sorted_cache=sorted_cache, sorted_in=False, sorted_out=True,
        x_raw_f16=h_mlp_f16, fused_swiglu=True, out_half=pure_nv_dn,
        gate_cap=gate_cap, up_cap=up_cap,
    )

    # 2. RHT-1152-only (100% skipped on pure-NVFP4 dn layers; skips NVFP4 experts on mixed layers)
    if pure_nv_dn:
        act_rot_s = raw_sorted
        act_raw_s = raw_sorted
    elif need_dn_raw:
        k_rht = make_rht1152_from_raw_kernel(in_half=False, out_half=True, also_raw_f16=True, skip_mode=2)
        outs = k_rht(
            inputs=[
                raw_sorted.reshape(-1), dn_signs_in.reshape(-1), _fk._paley_b_flat(1152),
                offsets.astype(mx.uint32), order.astype(mx.uint32),
                eids_flat.astype(mx.uint32), dn_side.exp_fid.astype(mx.uint32),
            ],
            grid=(288, n_slots, 1),
            threadgroup=(288, 1, 1),
            output_shapes=[(n_slots, 1152), (n_slots, 1152)],
            output_dtypes=[mx.float16, mx.float16],
        )
        act_rot_s, act_raw_s = outs[0], outs[1]
    else:
        skip_m = 1 if dynamic_routing else 0
        k_rht = make_rht1152_from_raw_kernel(in_half=False, out_half=True, also_raw_f16=False, skip_mode=skip_m)
        ins = [raw_sorted.reshape(-1), dn_signs_in.reshape(-1), _fk._paley_b_flat(1152)]
        if skip_m == 1:
            ins.append(offsets.astype(mx.uint32))
        act_rot_s = k_rht(
            inputs=ins,
            grid=(288, n_slots, 1),
            threadgroup=(288, 1, 1),
            output_shapes=[(n_slots, 1152)],
            output_dtypes=[mx.float16],
        )[0]
        act_raw_s = None

    # 3. DN projection reading coalesced sorted activations (`sorted_in=True`) and scattering `y_dn` to slot order
    return run_moe_side_prefill_v4(
        dn_side, act_rot_s, eids_flat, x_div=1,
        sorted_cache=sorted_cache, sorted_in=True, sorted_out=False,
        x_raw_f16=act_raw_s, fused_swiglu=False,
    )


@lru_cache(maxsize=4)
def make_swiglu_only_f16_kernel(skip_sentinel: bool = False):
    """Zero-RHT, zero-barrier `y_gu -> act_raw_f16` kernel for pure-NVFP4 `dn` layers during decode (`T <= 8`).

    On pure-NVFP4 `dn` visits (`dn.fids == (7,)`, 19/48 visits of 12.8GB), `dn_unified` only reads `x_raw_f16`
    and never touches `act_rot`. This kernel computes softcapped SwiGLU with `grid=(288, n_slots, 1)` and
    `threadgroup=(96, 1, 1)` (24 TGs at T=1, 144 TGs at T=6, 192 TGs at T=8 across 16 GPU cores), completely
    eliminating `H_16 + Paley-72` RHT-1152 and the second output buffer (`max_diff == 0.0`).
    """
    extra_in = ["eids"] if skip_sentinel else []
    guard = (
        """
    if (eids[slot] == 0xFFFFFFFFu) {
        ((device half4*)(raw_out + (ulong)slot * 1152u))[tid] = half4(0.0h);
        return;
    }"""
        if skip_sentinel
        else ""
    )
    src = f"""
    const uint tid = thread_position_in_grid.x;   // 0..287
    const uint slot = thread_position_in_grid.y;  // 0..n_slots-1
    const float g_cap = caps[0], g_inv = caps[1], u_cap = caps[2], u_inv = caps[3];{guard}
    const device float4* gp4 = (const device float4*)(y_gu + (ulong)slot * 2304u);
    const float4 g = gp4[tid], u = gp4[288u + tid];
    const float4 raw_act = (metal::precise::tanh(g * g_inv) * g_cap)
                         * (1.0f / (1.0f + metal::precise::exp(-g)))
                         * (metal::precise::tanh(u * u_inv) * u_cap);
    ((device half4*)(raw_out + (ulong)slot * 1152u))[tid] = half4(raw_act);
    """
    return mx.fast.metal_kernel(
        name=f"ink_swiglu_only_f16_sk{int(skip_sentinel)}",
        input_names=["y_gu", "caps"] + extra_in,
        output_names=["raw_out"],
        source=src,
    )


def fast_swiglu_only_f16(
    y_gu: mx.array,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
    *,
    eids_flat: mx.array | None = None,
) -> mx.array:
    """Compute `[n_slots, 1152] float16` softcapped SwiGLU without RHT-1152 (`max_diff == 0.0`)."""
    n_slots = int(y_gu.shape[0])
    caps = _fk._f32_arr(gate_cap, 1.0 / gate_cap, up_cap, 1.0 / up_cap)
    skip_s = eids_flat is not None
    kern = make_swiglu_only_f16_kernel(skip_sentinel=skip_s)
    ins = [y_gu.reshape(-1), caps]
    if skip_s:
        ins.append(eids_flat.astype(mx.uint32))
    return kern(
        inputs=ins,
        grid=(288, n_slots, 1),
        threadgroup=(96, 1, 1),
        output_shapes=[(n_slots, 1152)],
        output_dtypes=[mx.float16],
    )[0]


@lru_cache(maxsize=4)
def make_nv_skip_swiglu_rht1152_f16_kernel(use_sg8: bool = False):
    """1-slot-per-TG `SwiGLU + NVFP4-skipping RHT-1152` for mixed `dn` visits during decode (`T <= 8`).

    On the 23 mixed `dn` visits (`dn.has_nvfp4 and dn.fids != (7,)`), 75-90% of experts are NVFP4 (`fid == 7`).
    Since `slot = threadgroup_position_in_grid.y` is uniform across the threadgroup, any slot with
    `exp_fid[eids[slot]] == 7` writes `raw_out` and returns before barriers or Paley-72 `B` loads (`max_diff == 0.0`).
    """
    if not use_sg8:
        src_128 = """
        const uint slot = threadgroup_position_in_grid.y;
        const uint simd = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint hw = (simd << 1) | (lane >> 4); // 0..7
        const uint hl = lane & 15u;                // 0..15
        const uint c0 = hw * 9u;
        device half* rop = raw_out + (ulong)slot * 1152u;
        const uint eid = eids[slot];
        if (eid == 0xFFFFFFFFu) {
            for (uint j = 0; j < 9u; j++) rop[hl * 72u + c0 + j] = (half)0.0f;
            return;
        }
        const device float* gp = y_gu + (ulong)slot * 2304u;
        const device float* up = gp + 1152u;
        const float g_cap = caps[0], g_inv = caps[1], u_cap = caps[2], u_inv = caps[3];
        if (exp_fid[eid] == 7u) {
            #pragma unroll
            for (uint j = 0; j < 9u; j++) {
                const uint idx = hl * 72u + c0 + j;
                const float g = gp[idx], u = up[idx];
                const float act = (metal::precise::tanh(g * g_inv) * g_cap)
                                * (1.0f / (1.0f + metal::precise::exp(-g)))
                                * (metal::precise::tanh(u * u_inv) * u_cap);
                rop[idx] = (half)act;
            }
            return;
        }
        device half* op = out + (ulong)slot * 1152u;
        threadgroup float tg_z[16 * 72];
        const bool h1 = hl & 1u, h2 = hl & 2u, h4 = hl & 4u, h8 = hl & 8u;
        #pragma unroll
        for (uint j = 0; j < 9u; j++) {
            const uint c = c0 + j;
            const uint idx = hl * 72u + c;
            const float g = gp[idx], u = up[idx];
            const float act = (metal::precise::tanh(g * g_inv) * g_cap)
                            * (1.0f / (1.0f + metal::precise::exp(-g)))
                            * (metal::precise::tanh(u * u_inv) * u_cap);
            rop[idx] = (half)act;
            float v = act * signs[idx];
            float o;
            o = simd_shuffle_xor(v, 1u); v = h1 ? (o - v) : (v + o);
            o = simd_shuffle_xor(v, 2u); v = h2 ? (o - v) : (v + o);
            o = simd_shuffle_xor(v, 4u); v = h4 ? (o - v) : (v + o);
            o = simd_shuffle_xor(v, 8u); v = h8 ? (o - v) : (v + o);
            tg_z[idx] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup float* zr = tg_z + hl * 72u;
        constexpr float INV_S = 0.02946278254943948f;
        #pragma unroll
        for (uint j = 0; j < 9u; j++) {
            const uint c = c0 + j;
            const device float4* br = (const device float4*)(B + c * 72u);
            float acc = 0.0f;
            #pragma unroll
            for (uint k4 = 0; k4 < 18u; k4++) {
                const float4 b4 = br[k4];
                const uint k = k4 * 4u;
                acc += zr[k] * b4.x + zr[k + 1u] * b4.y + zr[k + 2u] * b4.z + zr[k + 3u] * b4.w;
            }
            op[hl * 72u + c] = (half)(acc * INV_S);
        }
        """
        return mx.fast.metal_kernel(
            name="ink_nv_skip_swiglu_rht1152_128_h",
            input_names=["y_gu", "signs", "B", "caps", "eids", "exp_fid"],
            output_names=["out", "raw_out"],
            source=src_128,
        )
    src = """
    const uint slot = threadgroup_position_in_grid.y;
    const uint simd = simdgroup_index_in_threadgroup; // 0..8 (9 simdgroups = 288 threads)
    const uint lane = thread_index_in_simdgroup;      // 0..31
    const uint tid = simd * 32u + lane;               // 0..287
    device half4* rop4 = (device half4*)(raw_out + (ulong)slot * 1152u);
    const uint eid = eids[slot];
    if (eid == 0xFFFFFFFFu) {
        rop4[tid] = half4(0.0h);
        return;
    }
    const device float4* gp4 = (const device float4*)(y_gu + (ulong)slot * 2304u);
    const float g_cap = caps[0], g_inv = caps[1], u_cap = caps[2], u_inv = caps[3];
    const float4 g = gp4[tid], u = gp4[288u + tid];
    const float4 raw_act = (metal::precise::tanh(g * g_inv) * g_cap)
                         * (1.0f / (1.0f + metal::precise::exp(-g)))
                         * (metal::precise::tanh(u * u_inv) * u_cap);
    rop4[tid] = half4(raw_act);
    if (exp_fid[eid] == 7u) {
        return;
    }

    const uint hw = lane >> 4, hl = lane & 15u;
    const float4 s4 = ((const device float4*)signs)[tid];
    const float4 act = raw_act * s4;
    threadgroup float tg_z[16 * 73];
    const uint a = tid / 18u, c = (tid - a * 18u) * 4u;
    threadgroup float* dst = tg_z + a * 73u + c;
    dst[0] = act.x; dst[1] = act.y; dst[2] = act.z; dst[3] = act.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c0 = simd * 8u + hw * 4u;
    const bool h1 = hl & 1u, h2 = hl & 2u, h4 = hl & 4u, h8 = hl & 8u;
    #pragma unroll
    for (uint j = 0; j < 4u; j++) {
        const uint cc = c0 + j;
        float v = tg_z[hl * 73u + cc];
        float o;
        o = simd_shuffle_xor(v, 1u); v = h1 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 2u); v = h2 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 4u); v = h4 ? (o - v) : (v + o);
        o = simd_shuffle_xor(v, 8u); v = h8 ? (o - v) : (v + o);
        tg_z[hl * 73u + cc] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 Acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 Acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma unroll
    for (uint kb = 0; kb < 9u; kb++) {
        simdgroup_float8x8 Bm, Zm0, Zm1;
        simdgroup_load(Bm, B + (simd * 8u) * 72u + kb * 8u, 72u, ulong2(0, 0), true);
        simdgroup_load(Zm0, tg_z + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_load(Zm1, tg_z + 8u * 73u + kb * 8u, 73u, ulong2(0, 0), false);
        simdgroup_multiply_accumulate(Acc0, Zm0, Bm, Acc0);
        simdgroup_multiply_accumulate(Acc1, Zm1, Bm, Acc1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(Acc0, tg_z + simd * 8u, 72u);
    simdgroup_store(Acc1, tg_z + 8u * 72u + simd * 8u, 72u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr float INV_S = 0.02946278254943948f;
    const threadgroup float4* tz4 = (const threadgroup float4*)tg_z;
    ((device half4*)(out + (ulong)slot * 1152u))[tid] = half4(tz4[tid] * INV_S);
    """
    return mx.fast.metal_kernel(
        name="ink_nv_skip_swiglu_rht1152_sg8_h",
        input_names=["y_gu", "signs", "B", "caps", "eids", "exp_fid"],
        output_names=["out", "raw_out"],
        header="#include <metal_simdgroup_matrix>\n",
        source=src,
    )


def fast_nv_skip_swiglu_rht1152_f16(
    y_gu: mx.array,
    signs: mx.array,
    eids_flat: mx.array,
    exp_fid: mx.array,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> tuple[mx.array, mx.array]:
    """Compute `(act_rot, act_raw)` for mixed NVFP4 `dn` layers at `T <= 8`, skipping RHT on NVFP4 slots."""
    n_slots = int(y_gu.shape[0])
    caps = _fk._f32_arr(gate_cap, 1.0 / gate_cap, up_cap, 1.0 / up_cap)
    use_sg8 = n_slots >= 64
    tg_x = 288 if use_sg8 else 128
    kern = make_nv_skip_swiglu_rht1152_f16_kernel(use_sg8=use_sg8)
    outs = kern(
        inputs=[
            y_gu.reshape(-1),
            signs.reshape(-1),
            _fk._paley_b_flat(1152),
            caps,
            eids_flat.astype(mx.uint32),
            exp_fid.astype(mx.uint32),
        ],
        grid=(tg_x, n_slots, 1),
        threadgroup=(tg_x, 1, 1),
        output_shapes=[(n_slots, 1152), (n_slots, 1152)],
        output_dtypes=[mx.float16, mx.float16],
    )
    return outs[0], outs[1]


