"""Numerical parity and O(N) KV allocation tests for custom Metal SDPA kernels & M4 Pro accelerations."""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from inkling_mlx_vision import accel
from runtime_mlx.inkling import drafter as _drafter_mod
from runtime_mlx.inkling import fast_kernels as _fk
from runtime_mlx.inkling import model as _model_mod


def _make_sdpa_inputs(
    T: int,
    kv_len: int,
    kv_mode: str,
    *,
    seed: int = 42,
    kv_start: int = 0,
    q_pos0: int | None = None,
    tree_parents: tuple[int, ...] | None = None,
):
    rng = np.random.default_rng(seed)
    kv_cap = ((kv_len + 255) // 256) * 256
    if q_pos0 is None:
        q_pos0 = kv_start + kv_len - T
    q = mx.array(rng.standard_normal((T, 24, 128)).astype(np.float32) * 0.5)
    rel_st = mx.array(rng.standard_normal((T, 24, 16)).astype(np.float32) * 0.25)
    rel_proj_t = mx.array(rng.standard_normal((1024, 16)).astype(np.float16) * 0.25)
    q_in = mx.array(rng.standard_normal((T, 2304)).astype(np.float32) * 0.1)
    sink_w = mx.array(rng.standard_normal((24, 2304)).astype(np.float16) * 0.05)

    kv_f16_raw = mx.array(rng.standard_normal((2, 8, kv_len, 128)).astype(np.float16) * 0.5)
    kv_q4_sc = None
    kv_fp8 = False
    if kv_mode == "fp16":
        kv_buf = mx.zeros((2, 8, kv_cap, 128), dtype=mx.float16)
        kv_buf[:, :, :kv_len, :] = kv_f16_raw
    elif kv_mode == "fp8":
        kv_fp8 = True
        kv_buf = mx.zeros((2, 8, kv_cap, 128), dtype=mx.uint8)
        kv_buf[:, :, :kv_len, :] = _fk.fp8_e4m3_quantize_f16_to_u8(kv_f16_raw)
    elif kv_mode == "q4":
        q4_u8, q4_sc = _fk.kv_h128_q4_quantize(kv_f16_raw)
        kv_buf = mx.zeros((2, 8, kv_cap, 64), dtype=mx.uint8)
        kv_q4_sc = mx.zeros((2, 8, kv_cap, 4), dtype=mx.float16)
        kv_buf[:, :, :kv_len, :] = q4_u8
        kv_q4_sc[:, :, :kv_len, :] = q4_sc
    else:
        raise ValueError(kv_mode)

    tree_depth = None
    tree_mask = None
    if tree_parents is not None:
        spec = _fk.make_tree_spec(tree_parents)
        tree_depth = spec["depth_mx"]
        tree_mask = spec["mask_mx"]
        q_pos0 = kv_start + kv_len - T

    mx.eval(q, rel_st, rel_proj_t, q_in, sink_w, kv_buf)
    if kv_q4_sc is not None:
        mx.eval(kv_q4_sc)
    return dict(
        q=q,
        KV_buf=kv_buf,
        rel_st=rel_st,
        rel_proj_t_f16=rel_proj_t,
        q_in=q_in,
        sink_w_f16=sink_w,
        T=T,
        kv_len=kv_len,
        kv_start=kv_start,
        q_pos0=q_pos0,
        kv_cap=kv_cap,
        is_sliding=False,
        tree_depth=tree_depth,
        tree_mask=tree_mask,
        kv_fp8=kv_fp8,
        kv_q4_scales=kv_q4_sc,
    )


def _assert_close(got: mx.array, ref: mx.array, *, max_tol: float = 5e-4, rmse_tol: float = 5e-5):
    mx.eval(got, ref)
    g = np.asarray(got, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    assert g.shape == r.shape
    assert np.all(np.isfinite(g)), "non-finite values in accelerated SDPA output"
    diff = np.abs(g - r)
    max_err = float(diff.max())
    rmse = float(np.sqrt(np.mean(diff * diff)))
    assert max_err < max_tol and rmse < rmse_tol, f"max_err={max_err:.6g} (tol {max_tol}), rmse={rmse:.6g} (tol {rmse_tol})"


@pytest.mark.parametrize("kv_mode", ["fp16", "fp8", "q4"])
@pytest.mark.parametrize("T", [2, 3, 5, 7, 8])
def test_gqa_sg8_splitk_causal_parity(kv_mode: str, T: int):
    # Test both medium (kv_len=2053, exercises boundary + fast_interior + partial tail) and 8K+ splits
    for kv_len in (2053, 8211):
        kw = _make_sdpa_inputs(T, kv_len, kv_mode, seed=100 + T + kv_len)
        ref = accel._ORIG_GQA_SDPA(**kw)
        got = accel.accelerated_gqa_sdpa(**kw)
        _assert_close(got, ref)


@pytest.mark.parametrize("kv_mode", ["fp16", "fp8", "q4"])
@pytest.mark.parametrize(
    "parents",
    [
        (-1, 0, 1, 0),  # 2-1 (T=4)
        (-1, 0, 1, 2, 0, 4),  # 3-2 (T=6)
        (-1, 0, 1, 2, 3, 0, 5, 6),  # 4-3 (T=8)
        (-1, 0, 1, 2, 3, 4, 0, 6),  # 5-2 (T=8)
    ],
)
def test_gqa_sg8_splitk_tree_parity(kv_mode: str, parents: tuple[int, ...]):
    T = len(parents)
    for kv_len in (2055, 8207):
        kw = _make_sdpa_inputs(T, kv_len, kv_mode, seed=200 + T + kv_len, tree_parents=parents)
        ref = accel._ORIG_GQA_SDPA(**kw)
        got = accel.accelerated_gqa_sdpa(**kw)
        _assert_close(got, ref)


@pytest.mark.parametrize("kv_mode", ["fp16", "fp8", "q4"])
@pytest.mark.parametrize("T", [32, 33, 64, 65, 256])
def test_gqa_shared_prefill_fa2_parity(kv_mode: str, T: int):
    # Exercises both q_tiles=2 (T=32,33) and q_tiles=4 (T=64,65,256), plus non-multiple-of-32 tails
    kv_len = 4129 + T
    kw = _make_sdpa_inputs(T, kv_len, kv_mode, seed=300 + T)
    ref = accel._ORIG_GQA_SDPA(**kw)
    got = accel.accelerated_gqa_sdpa(**kw)
    _assert_close(got, ref)


@pytest.mark.parametrize("kv_mode", ["fp16", "fp8", "q4"])
def test_t1_splitk48_parity(kv_mode: str):
    kw = _make_sdpa_inputs(1, 8213, kv_mode, seed=400)
    ref = accel._ORIG_GQA_SDPA(**kw)
    got = accel.accelerated_gqa_sdpa(**kw)
    _assert_close(got, ref, max_tol=1e-4, rmse_tol=1e-5)


def test_sliding_fallback_parity():
    kw = _make_sdpa_inputs(8, 1280, "fp16", seed=500)
    kw["is_sliding"] = True
    kw["kv_start"] = 2000
    kw["q_pos0"] = 2000 + 1280 - 8
    ref = accel._ORIG_GQA_SDPA(**kw)
    got = accel.accelerated_gqa_sdpa(**kw)
    _assert_close(got, ref, max_tol=1e-6, rmse_tol=1e-6)


def test_install_and_kv_reserve_cap():
    logs: list[str] = []
    info = accel.install_metal_accelerations(log_fn=logs.append)
    assert info["installed"] is True
    assert _fk.gqa_relpos_sinks_sdpa_fast is accel.accelerated_gqa_sdpa
    assert _model_mod.gqa_relpos_sinks_sdpa_fast is accel.accelerated_gqa_sdpa

    # Verify Inkling2Cache pre-allocation via _reserve_cap and 4096-step growth
    cfg = _model_mod.Inkling2Config()
    cfg.global_kv_mode = "fp8"
    cache = _model_mod.Inkling2Cache(cfg)
    cache._reserve_cap = 6000  # rounds up to 8192 (multiple of 4096)

    chunk = mx.zeros((2, 8, 256, 128), dtype=mx.float16)
    buf0, kv_len, _, kv_cap, is_fp8, _ = cache.update_kv_f16(0, chunk, is_sliding=False)
    assert is_fp8 is True
    assert kv_len == 256
    assert kv_cap == 8192

    # Subsequent appends up to 8192 must NOT reallocate
    for step in range(1, 10):
        buf_next, kv_len, _, kv_cap, _, _ = cache.update_kv_f16(0, chunk, is_sliding=False)
        assert kv_cap == 8192
        assert kv_len == (step + 1) * 256
        assert buf_next is buf0

    # Sliding layers keep tight 256-step allocation (must NOT blow up to 8192)
    _, s_len, _, s_cap, s_fp8, _ = cache.update_kv_f16(1, chunk, is_sliding=True)
    assert s_fp8 is False
    assert s_len == 256
    assert s_cap == 256

    # Q4 global KV pre-allocation via _reserve_cap (both kv_u8 and kv_q4_sc)
    cfg_q4 = _model_mod.Inkling2Config()
    cfg_q4.global_kv_mode = "q4"
    cache_q4 = _model_mod.Inkling2Cache(cfg_q4)
    cache_q4._reserve_cap = 5000
    q4_buf, q4_len, _, q4_cap, q4_fp8, q4_sc = cache_q4.update_kv_f16(0, chunk, is_sliding=False)
    assert q4_fp8 is False and q4_sc is not None
    assert q4_len == 256 and q4_cap == 8192 and q4_sc.shape[2] == 8192
    q4_buf2, _, _, q4_cap2, _, q4_sc2 = cache_q4.update_kv_f16(0, chunk, is_sliding=False)
    assert q4_cap2 == 8192 and q4_buf2 is q4_buf and q4_sc2 is q4_sc

    # Drafter global vs local _ensure_ctx_cap
    class _FakeDrafter:
        num_kv_heads = 2
        head_dim = 128
        local_layers = {0}
        ctx_k = [None, None]
        ctx_v = [None, None]
        _ctx_cur = [0, 0]
        _ctx_cap = [0, 0]
        _reserve_cap = 5000

    fd = _FakeDrafter()
    _drafter_mod.DSparkDrafterMLX._ensure_ctx_cap(fd, 0, 300)  # local layer -> 512
    _drafter_mod.DSparkDrafterMLX._ensure_ctx_cap(fd, 1, 300)  # global layer -> 8192
    assert fd._ctx_cap[0] == 512
    assert fd._ctx_cap[1] == 8192
    assert info["custom_drafter_gqa12_sdpa"] is True
    assert info["custom_radix_topp"] is True
    assert info["ngram_default"] is True
    assert _model_mod.logits_to_probs is accel.fast_logits_to_probs


@pytest.mark.parametrize("T", [2, 5, 6, 8])
@pytest.mark.parametrize(
    "lo,hi,kv_cap",
    [
        (0, 1031, 1280),       # local layer from 0
        (517, 1549, 2048),     # local layer with non-zero sliding window lo
        (0, 8219, 8448),       # global layer with 32 splits + partial tail
    ],
)
def test_drafter_gqa12_sdpa_parity(T: int, lo: int, hi: int, kv_cap: int):
    rng = np.random.default_rng(700 + T + lo + hi)
    q = mx.array((rng.standard_normal((1, 24, T, 128)) * 0.5).astype(np.float16))
    K = mx.array((rng.standard_normal((1, 2, kv_cap, 128)) * 0.5).astype(np.float16))
    V = mx.array((rng.standard_normal((1, 2, kv_cap, 128)) * 0.5).astype(np.float16))
    mx.eval(q, K, V)

    ref = mx.fast.scaled_dot_product_attention(q, K[:, :, lo:hi, :], V[:, :, lo:hi, :], scale=128.0 ** -0.5)
    got = accel.drafter_gqa_sdpa_fast(q, K, V, lo, hi, kv_cap)
    assert got.dtype == mx.float16
    _assert_close(got, ref, max_tol=5e-4, rmse_tol=5e-5)


@pytest.mark.parametrize(
    "shape,scale,temp,top_p",
    [
        ((8, 200058), 5.0, 0.6, 0.95),     # real Inkling-2 200K vocab (V % 4 == 2, tests unaligned rows!)
        ((8, 200058), 0.2, 1.0, 0.95),     # ultra-flat 200K distribution
        ((32768,), 4.0, 0.6, 0.95),        # 1D draft shortlist
        ((5, 2003), 3.0, 0.7, 0.90),       # odd V tail handling
        ((4, 4096), 5.0, 0.6, 0.05),       # single-token Pass-1 early exit
        ((4, 4096), 2.0, 1.0, 0.999999),   # float-rounding boundary fallback
    ],
)
def test_exact_radix_topp_parity(shape: tuple[int, ...], scale: float, temp: float, top_p: float):
    rng = np.random.default_rng(900 + sum(shape) + int(top_p * 1000))
    x = mx.array((rng.standard_normal(shape) * scale).astype(np.float32))
    mx.eval(x)

    ref = accel._ORIG_LOGITS_TO_PROBS(x, temp=temp, top_p=top_p, top_k=0)
    got = accel.fast_logits_to_probs(x, temp=temp, top_p=top_p, top_k=0)
    mx.eval(ref, got)
    g = np.asarray(got, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    assert g.shape == r.shape
    assert np.all(np.isfinite(g)) and np.all(g >= 0.0)
    row_sums = g.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)
    diff = np.abs(g - r)
    assert float(diff.max()) < 1e-5
    assert float(np.sqrt(np.mean(diff * diff))) < 1e-7


@pytest.mark.parametrize("sg", [8, 4])
def test_fast_moe_prefill_v3_parity(sg: int):
    """Bit-exact parity of Lean-SRAM Union-LUT MoE Prefill v3 across nb_val in (1..6) and fid in (0, 3)."""
    rng = np.random.default_rng(1234 + sg)
    # 6 experts exercising cnt in {1..8}, {9..16}, {17..24}, {25..32}, {33..40}, {41..48} and both HT (fid=0) & RTR (fid=3)
    counts = [6, 15, 21, 29, 37, 45]
    n_slots = sum(counts)
    n_cols = 256
    m_rows = 128

    x_rot = mx.array((rng.standard_normal((n_slots, n_cols)) * 0.5).astype(np.float16))
    codes = mx.array(rng.integers(0, 65536, size=(98304,), dtype=np.uint32))
    scales = mx.array((rng.uniform(0.05, 0.5, size=(24576,))).astype(np.float16))
    exp_fid = mx.array([0, 3, 0, 3, 0, 3], dtype=mx.uint8)
    exp_code_off = mx.array([i * 8192 for i in range(6)], dtype=mx.uint32)
    exp_scale_off = mx.array([i * 2048 for i in range(6)], dtype=mx.uint32)
    lut_ht1 = mx.array((rng.standard_normal((512, 4)) * 0.25).astype(np.float16))
    lut_ht2 = mx.array((rng.standard_normal((512, 4)) * 0.25).astype(np.float16))
    lut_ht3 = mx.array((rng.standard_normal((512, 4)) * 0.25).astype(np.float16))
    lut_rtr1 = mx.array((rng.standard_normal((256, 16)) * 0.25).astype(np.float16))
    lut_rtr2 = mx.array((rng.standard_normal((256, 16)) * 0.25).astype(np.float16))
    lut_rtr3 = mx.array((rng.standard_normal((256, 16)) * 0.25).astype(np.float16))
    lut_rtr4 = mx.array((rng.standard_normal((65536,)) * 0.25).astype(np.float16))

    ref_units_list = []
    got_units_list = []
    cur = 0
    for e, c in enumerate(counts):
        got_units_list.extend([e, cur, c])
        rem = c
        off = cur
        while rem > 0:
            step_c = min(rem, 32)
            ref_units_list.extend([e, off, step_c])
            off += step_c
            rem -= step_c
        cur += c
    ref_units = mx.array(ref_units_list, dtype=mx.uint32)
    ref_ucnt = mx.array([len(ref_units_list) // 3], dtype=mx.uint32)
    got_units = mx.array(got_units_list, dtype=mx.uint32)
    got_ucnt = mx.array([len(got_units_list) // 3], dtype=mx.uint32)
    meta = _fk._u32(m_rows, 8)
    slot_order = mx.arange(n_slots, dtype=mx.uint32)

    common = [
        x_rot.reshape(-1), codes, scales, exp_fid, exp_code_off, exp_scale_off,
        lut_ht1, lut_ht2, lut_ht3, lut_rtr1, lut_rtr2, lut_rtr3, lut_rtr4,
    ]
    mx.eval(*common, ref_units, ref_ucnt, got_units, got_ucnt, meta, slot_order)

    fids = (0, 3)
    # Stock v2 reference uses sg=8, tile_t=32
    ref_k = _fk._make_unified_moe_prefill_v2_kernel(n_cols, fids, sg=8, rb=2)
    got_k = accel.make_fast_moe_prefill_v3_kernel(n_cols, fids, sg=sg, rb=2, max_nb=6)
    ref_max_u = len(ref_units_list) // 3
    got_max_u = len(got_units_list) // 3
    ref_rblocks = (m_rows + 8 * 8 * 2 - 1) // (8 * 8 * 2)
    got_rblocks = (m_rows + sg * 8 * 2 - 1) // (sg * 8 * 2)

    ref = ref_k(
        inputs=[*common, ref_units, ref_ucnt, meta, slot_order],
        grid=(ref_max_u * 8 * 32, ref_rblocks, 1),
        threadgroup=(8 * 32, 1, 1),
        output_shapes=[(n_slots, m_rows)],
        output_dtypes=[mx.float32],
    )[0]
    got = got_k(
        inputs=[*common, got_units, got_ucnt, meta, slot_order],
        grid=(got_max_u * sg * 32, got_rblocks, 1),
        threadgroup=(sg * 32, 1, 1),
        output_shapes=[(n_slots, m_rows)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(ref, got)
    diff = np.max(np.abs(np.asarray(ref, dtype=np.float32) - np.asarray(got, dtype=np.float32)))
    assert diff == 0.0, f"MoE prefill v3 (sg={sg}, max_nb=6) max_diff={diff}"

