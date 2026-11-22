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
