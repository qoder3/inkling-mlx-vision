"""Bit-exact parity tests for MoE Prefill v4 (`tile_t=64`, per-j A0/A1 streaming, fused GU->SwiGLU, NVFP4-skipping RHT-1152)."""

from __future__ import annotations

import types
import numpy as np
import pytest
import mlx.core as mx

from inkling_mlx_vision import accel, moe_v4
from runtime_mlx.inkling import fast_kernels as _fk


def _make_luts(rng: np.random.Generator) -> tuple[list[mx.array], mx.array, mx.array]:
    lut_ht1 = mx.array((rng.standard_normal((512, 4)) * 0.25).astype(np.float16))
    lut_ht2 = mx.array((rng.standard_normal((512, 4)) * 0.25).astype(np.float16))
    lut_ht3 = mx.array((rng.standard_normal((512, 4)) * 0.25).astype(np.float16))
    lut_rtr1 = mx.array((rng.standard_normal((256, 16)) * 0.25).astype(np.float16))
    lut_rtr2 = mx.array((rng.standard_normal((256, 16)) * 0.25).astype(np.float16))
    lut_rtr3 = mx.array((rng.standard_normal((256, 16)) * 0.25).astype(np.float16))
    lut_rtr4 = mx.array((rng.standard_normal((65536,)) * 0.25).astype(np.float16))
    lut_e2m1 = mx.array((rng.standard_normal((16,)) * 0.5).astype(np.float16))
    lut_e4m3 = mx.array(rng.uniform(0.05, 0.5, size=(256,)).astype(np.float16))
    luts = [lut_ht1, lut_ht2, lut_ht3, lut_rtr1, lut_rtr2, lut_rtr3, lut_rtr4]
    mx.eval(*luts, lut_e2m1, lut_e4m3)
    return luts, lut_e2m1, lut_e4m3


@pytest.mark.parametrize("single_buf", [False, True])
def test_moe_v4_all_8_formats_unfused_bit_exact(single_buf: bool):
    """Verify bit-exact parity (`max_diff == 0.0`) across ALL 8 formats (fid=0..7) and ALL nb=1..8."""
    rng = np.random.default_rng(701 + int(single_buf))
    fids = (0, 1, 2, 3, 4, 5, 6, 7)
    # 8 experts: one per format fid=0..7, counts covering nb=1..8 (both unstreamed 1..4 and j-streamed 5..8)!
    counts = [5, 13, 21, 29, 37, 45, 53, 61]
    n_slots = sum(counts)
    n_cols = 256
    m_rows = 128

    x_rot = mx.array((rng.standard_normal((n_slots, n_cols)) * 0.3).astype(np.float16))
    x_raw = mx.array((rng.standard_normal((n_slots, n_cols)) * 0.3).astype(np.float16))
    luts, lut_e2m1, lut_e4m3 = _make_luts(rng)

    # Pack 1 slab per fid
    code_parts, scale_parts = [], []
    exp_code_off, exp_scale_off = [], []
    c_off = s_off = 0
    for fid in fids:
        if fid == 7:
            wpr = n_cols // 8
            c = rng.integers(0, 1 << 32, size=(m_rows * wpr,), dtype=np.uint32)
            s = np.zeros((64,), dtype=np.float16)
        else:
            k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
            wpr = (n_cols * k) // 32
            c = rng.integers(0, 1 << 32, size=(m_rows * wpr,), dtype=np.uint32)
            s = rng.uniform(0.05, 0.3, size=(m_rows,)).astype(np.float16)
        exp_code_off.append(c_off)
        exp_scale_off.append(s_off)
        code_parts.append(c)
        scale_parts.append(s)
        c_off += c.size
        s_off += s.size
    code_parts.append(np.zeros(8, dtype=np.uint32))
    scale_parts.append(np.zeros(64, dtype=np.float16))

    codes = mx.array(np.concatenate(code_parts))
    scales = mx.array(np.concatenate(scale_parts))
    exp_fid = mx.array(np.array(fids, dtype=np.uint32))
    code_off_mx = mx.array(np.array(exp_code_off, dtype=np.uint32))
    scale_off_mx = mx.array(np.array(exp_scale_off, dtype=np.uint32))
    nv_scales = mx.array(rng.integers(0, 256, size=(m_rows * (n_cols // 16) + 64,), dtype=np.uint8))
    nv_soff = mx.zeros((8,), dtype=mx.uint32)
    nv_outer = mx.full((8,), 0.25, dtype=mx.float32)

    ref_units, got_units = [], []
    cur = 0
    for e, cnt in enumerate(counts):
        got_units.extend([e, cur, cnt])
        rem, off = cnt, cur
        while rem > 0:
            step = min(rem, 32)
            ref_units.extend([e, off, step])
            off += step
            rem -= step
        cur += cnt
    ref_u = mx.array(ref_units, dtype=mx.uint32)
    ref_uc = mx.array([len(ref_units) // 3], dtype=mx.uint32)
    got_u = mx.array(got_units, dtype=mx.uint32)
    got_uc = mx.array([len(got_units) // 3], dtype=mx.uint32)
    meta = _fk._u32(m_rows, 1)
    slot_order = mx.arange(n_slots, dtype=mx.uint32)
    nv_tail = [x_raw.reshape(-1), nv_scales, nv_soff, nv_outer, lut_e2m1, lut_e4m3]
    common = [x_rot.reshape(-1), codes, scales, exp_fid, code_off_mx, scale_off_mx, *luts]
    mx.eval(*common, *nv_tail, ref_u, ref_uc, got_u, got_uc, meta, slot_order)

    ref_k = _fk._make_unified_moe_prefill_v2_kernel(n_cols, fids, sg=8, rb=2)
    got_k = moe_v4.make_fast_moe_prefill_v4_kernel(n_cols, fids, sg=4, rb=2, max_nb=8, single_buf=single_buf)
    ref = ref_k(
        inputs=[*common, ref_u, ref_uc, meta, slot_order, *nv_tail],
        grid=((len(ref_units) // 3) * 8 * 32, (m_rows + 127) // 128, 1),
        threadgroup=(8 * 32, 1, 1),
        output_shapes=[(n_slots, m_rows)],
        output_dtypes=[mx.float32],
    )[0]
    got = got_k(
        inputs=[*common, got_u, got_uc, meta, slot_order, *nv_tail],
        grid=((len(got_units) // 3) * 4 * 32, (m_rows + 63) // 64, 1),
        threadgroup=(4 * 32, 1, 1),
        output_shapes=[(n_slots, m_rows)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(ref, got)
    diff = float(np.max(np.abs(np.asarray(ref, dtype=np.float32) - np.asarray(got, dtype=np.float32))))
    assert diff == 0.0, f"MoE v4 all-8-format parity failed (single_buf={single_buf}): max_diff={diff}"


def _make_mock_side(rng: np.random.Generator, m_rows: int, n_cols: int, exp_fids_128: np.ndarray, luts, lut_e2m1, lut_e4m3):
    fids = tuple(sorted(set(int(f) for f in exp_fids_128)))
    has_nv = 7 in fids
    n_phys = 2
    code_parts, scale_parts = [], []
    fid_cbase, fid_sbase, fid_wpr = {}, {}, {}
    c_off = s_off = 0
    for fid in fids:
        fid_cbase[fid] = c_off
        fid_sbase[fid] = s_off
        if fid == 7:
            wpr = n_cols // 8
            c = rng.integers(0, 1 << 32, size=(n_phys * m_rows * wpr,), dtype=np.uint32)
            s = np.zeros((64,), dtype=np.float16)
        else:
            k = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4}[fid]
            wpr = (n_cols * k) // 32
            c = rng.integers(0, 1 << 32, size=(n_phys * m_rows * wpr,), dtype=np.uint32)
            s = rng.uniform(0.05, 0.25, size=(n_phys * m_rows,)).astype(np.float16)
        fid_wpr[fid] = wpr
        code_parts.append(c)
        scale_parts.append(s)
        c_off += c.size
        s_off += s.size
    code_parts.append(np.zeros(8, dtype=np.uint32))
    scale_parts.append(np.zeros(64, dtype=np.float16))

    spr_nv = n_cols // 16
    exp_code_off = np.zeros(128, dtype=np.uint32)
    exp_scale_off = np.zeros(128, dtype=np.uint32)
    exp_nv_soff = np.zeros(128, dtype=np.uint32)
    for e in range(128):
        f = int(exp_fids_128[e])
        loc = e % n_phys
        exp_code_off[e] = fid_cbase[f] + loc * m_rows * fid_wpr[f]
        if f == 7:
            exp_scale_off[e] = fid_sbase[7]
            exp_nv_soff[e] = loc * m_rows * spr_nv
        else:
            exp_scale_off[e] = fid_sbase[f] + loc * m_rows

    side = types.SimpleNamespace()
    side.m_rows = m_rows
    side.n_cols = n_cols
    side.fids = fids
    side.n_rg = 1
    side.has_nvfp4 = has_nv
    side.codes = mx.array(np.concatenate(code_parts))
    side.scales = mx.array(np.concatenate(scale_parts))
    side.exp_fid = mx.array(exp_fids_128.astype(np.uint32))
    side._kern_fid = side.exp_fid
    side.exp_code_off = mx.array(exp_code_off)
    side.exp_scale_off = mx.array(exp_scale_off)
    side._lut_args_prefill_v2 = luts
    if has_nv:
        nv_scales = mx.array(rng.integers(0, 256, size=(n_phys * m_rows * spr_nv + 64,), dtype=np.uint8))
        nv_soff = mx.array(exp_nv_soff)
        nv_outer = mx.full((128,), 0.2, dtype=mx.float32)
        side._nv_args = [nv_scales, nv_soff, nv_outer, lut_e2m1, lut_e4m3]
        mx.eval(nv_scales, nv_soff, nv_outer)
    else:
        side._nv_args = []
    mx.eval(side.codes, side.scales, side.exp_fid, side.exp_code_off, side.exp_scale_off)
    return side


@pytest.mark.parametrize("dn_mode", ["pure_nv", "mixed_ht_nv", "no_nv"])
@pytest.mark.parametrize("dynamic_routing", [False, True])
def test_moe_v4_fused_pipeline_bit_exact(dn_mode: str, dynamic_routing: bool):
    """End-to-end `fused_moe_prefill_v4` (`GU -> SwiGLU -> [NVFP4-skip RHT-1152] -> DN`) bit-exact vs stock v2."""
    rng = np.random.default_rng(800 + hash((dn_mode, dynamic_routing)) % 1000)
    luts, lut_e2m1, lut_e4m3 = _make_luts(rng)

    # GU: mix of ht2(1), ht3(2), rtr4(6), nvfp4(7); m_rows=2304, n_cols=256 (fast small-K test)
    n_gu = 256
    gu_fids = np.array([[1, 2, 6, 7][e % 4] for e in range(128)], dtype=np.uint32)
    gu_side = _make_mock_side(rng, 2304, n_gu, gu_fids, luts, lut_e2m1, lut_e4m3)

    if dn_mode == "pure_nv":
        dn_fids = np.full(128, 7, dtype=np.uint32)
    elif dn_mode == "mixed_ht_nv":
        dn_fids = np.array([2 if e < 64 else 7 for e in range(128)], dtype=np.uint32)
    else:
        dn_fids = np.array([1 if e < 64 else 2 for e in range(128)], dtype=np.uint32)
    dn_side = _make_mock_side(rng, 256, 1152, dn_fids, luts, lut_e2m1, lut_e4m3)

    T = 80  # n_slots = 640 (> 512 -> triggers max_nb=6)
    n_slots = T * 8
    eids_np = np.concatenate([rng.choice(128, size=8, replace=False) for _ in range(T)]).astype(np.uint32)
    if dynamic_routing:
        # Mark ~20% of slots as sentinel 0xFFFFFFFF
        mask = rng.uniform(size=(n_slots,)) < 0.2
        eids_np[mask] = 0xFFFFFFFF
    eids = mx.array(eids_np)
    h_mlp = mx.array((rng.standard_normal((T, n_gu)) * 0.25).astype(np.float16))
    x_rot = mx.array((rng.standard_normal((T, n_gu)) * 0.25).astype(np.float16))
    signs = mx.array(np.where(rng.uniform(size=(1152,)) > 0.5, 1.0, -1.0).astype(np.float32))
    sc = _fk.compute_moe_sorted_offsets(eids)
    mx.eval(eids, h_mlp, x_rot, signs, sc[0], sc[1])

    # Stock reference: unfused v2 GU -> _ORIG_SWIGLU_RHT1152 -> unfused v2 DN
    def _run_v2(side, x_r, x_div, x_raw):
        u, uc, mu = _fk.plan_moe_prefill_units(sc[0], side.exp_fid, side.fids, n_slots, tile_t=32)
        k = _fk._make_unified_moe_prefill_v2_kernel(side.n_cols, side.fids, sg=8, rb=2)
        nv_t = [(x_raw if x_raw is not None else x_r).reshape(-1), *side._nv_args] if side.has_nvfp4 else []
        return k(
            inputs=[x_r.reshape(-1), side.codes, side.scales, side._kern_fid, side.exp_code_off, side.exp_scale_off, *luts, u, uc, _fk._u32(side.m_rows, x_div), sc[1], *nv_t],
            grid=(mu * 8 * 32, (side.m_rows + 127) // 128, 1),
            threadgroup=(8 * 32, 1, 1),
            output_shapes=[(n_slots, side.m_rows)],
            output_dtypes=[mx.float32],
        )[0]

    y_gu_ref = _run_v2(gu_side, x_rot, 8, h_mlp)
    sw_kw = {"eids_flat": eids} if dynamic_routing else {}
    if dn_side.has_nvfp4:
        act_rot_ref, act_raw_ref = accel._ORIG_SWIGLU_RHT1152(y_gu_ref, signs, 4.0, 25.0, out_dtype=mx.float16, also_raw=True, **sw_kw)
    else:
        act_rot_ref = accel._ORIG_SWIGLU_RHT1152(y_gu_ref, signs, 4.0, 25.0, out_dtype=mx.float16, **sw_kw)
        act_raw_ref = None
    ref = _run_v2(dn_side, act_rot_ref, 1, act_raw_ref)

    got = moe_v4.fused_moe_prefill_v4(
        gu_side, dn_side, h_mlp, x_rot, signs, eids, sc,
        gate_cap=4.0, up_cap=25.0, dynamic_routing=dynamic_routing,
    )
    mx.eval(ref, got)

    valid = (eids_np != 0xFFFFFFFF)[:, None]
    r_np = np.asarray(ref, dtype=np.float32)
    g_np = np.asarray(got, dtype=np.float32)
    diff = float(np.max(np.where(valid, np.abs(r_np - g_np), 0.0)))
    assert diff == 0.0, f"fused_moe_prefill_v4 parity failed ({dn_mode}, dyn={dynamic_routing}): max_diff={diff}"
