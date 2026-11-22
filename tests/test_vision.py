"""Unit tests for pure-MLX Inkling-2 Vision preprocessing, geometry, SSRF guard, and embedding splice."""

from __future__ import annotations

import base64
import io
import math
import types

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from inkling_mlx_vision import vision as iv


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_rgba_checkerboard_and_padding():
    # 16x16 fully transparent RGBA image -> must composite onto 8x8 (204/153) checkerboard,
    # then pad bottom-right to 40x40 with -1.0 before rescale/normalize.
    raw = _png_bytes(Image.new("RGBA", (16, 16), (0, 0, 0, 0)))
    prep = iv.preprocess_image(raw)
    assert (prep.rows, prep.cols) == (1, 1)
    assert prep.num_tokens == 1
    assert prep.pixel_values.shape == (1, 40, 40, 3)

    patch = prep.pixel_values[0]
    exp_light = (204.0 / 255.0 - 0.5) / 0.5
    exp_dark = (153.0 / 255.0 - 0.5) / 0.5
    exp_pad = (-1.0 / 255.0 - 0.5) / 0.5

    assert np.allclose(patch[0, 0], exp_light, atol=1e-5)
    assert np.allclose(patch[0, 8], exp_dark, atol=1e-5)
    assert np.allclose(patch[8, 0], exp_dark, atol=1e-5)
    assert np.allclose(patch[8, 8], exp_light, atol=1e-5)
    assert np.allclose(patch[20, 20], exp_pad, atol=1e-5)


def test_preprocess_downscales_extreme_4k_preserves_1080p():
    raw_1080p = _png_bytes(Image.new("RGB", (1920, 1080), (128, 128, 128)))
    p1 = iv.preprocess_image(raw_1080p)
    assert p1.proc_size == (1920, 1080)
    assert (p1.rows, p1.cols) == (27, 48)

    raw_4k = _png_bytes(Image.new("RGB", (3840, 2160), (128, 128, 128)))
    p2 = iv.preprocess_image(raw_4k)
    assert p2.orig_size == (3840, 2160)
    assert p2.proc_size[0] * p2.proc_size[1] <= iv.DEFAULT_MAX_PIXELS + 5000
    assert abs((p2.proc_size[0] / p2.proc_size[1]) - (16 / 9)) < 0.02


def test_bilinear_pos_interp_weights_sum_to_one():
    idx, weights = iv._pos_interp_indices_and_weights(rows=6, cols=10)
    assert idx.shape == (240, 4)
    assert weights.shape == (240, 4)
    w_np = np.array(weights)
    assert np.all(w_np >= 0.0)
    assert np.allclose(w_np.sum(axis=-1), 1.0, atol=1e-5)
    idx_np = np.array(idx)
    assert idx_np.min() >= 0 and idx_np.max() < iv.POSEMB_GRID_SIZE**2


def test_axial_rope_cls_zero_and_width_first_order():
    cos, sin = iv._axial_rope_cos_sin(rows=2, cols=3)
    # 1 CLS + 4*2*3 = 25 tokens
    assert cos.shape == (25, 1, 64)
    assert sin.shape == (25, 1, 64)
    # CLS token sits at (0, 0) -> cos == 1, sin == 0
    assert np.allclose(np.array(cos[0]), 1.0, atol=1e-6)
    assert np.allclose(np.array(sin[0]), 0.0, atol=1e-6)
    # Verify pairwise repeat: even/odd entries in each pair match
    cos_np = np.array(cos[1, 0])
    assert np.allclose(cos_np[0::2], cos_np[1::2], atol=1e-6)
    # Patch index 1 is (row=0, col=1) -> 1-indexed (h=1, w=2); width occupies dims 0..31, height dims 32..63
    # At freq 0 (inv_freq=1.0), w=2 -> cos(2), h=1 -> cos(1)
    assert math.isclose(float(cos[2, 0, 0]), math.cos(2.0), rel_tol=1e-5)
    assert math.isclose(float(cos[2, 0, 32]), math.cos(1.0), rel_tol=1e-5)


def test_ssrf_and_scheme_guards():
    raw = _png_bytes(Image.new("RGB", (40, 40), (10, 20, 30)))
    b64 = base64.b64encode(raw).decode()
    part_data = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    got, h = iv.load_image_bytes(part_data)
    assert got == raw and len(h) == 32

    # Remote disabled by default
    with pytest.raises(ValueError, match="disabled"):
        iv.load_image_bytes({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}})

    # Loopback / private IPs blocked even when allow_remote=True
    with pytest.raises(ValueError, match="non-public IP"):
        iv.load_image_bytes(
            {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8080/secret.png"}},
            allow_remote=True,
        )
    with pytest.raises(ValueError, match="non-public IP"):
        iv.load_image_bytes(
            {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest/meta-data/"}},
            allow_remote=True,
        )

    # file:// blocked when allow_file=False
    with pytest.raises(ValueError, match="loopback"):
        iv.load_image_bytes(
            {"type": "image_url", "image_url": {"url": "file:///etc/hosts"}},
            allow_file=False,
        )


def test_vision_aware_embed_multichunk_and_multiimage_splice():
    # Dummy text embedding always returns zeros [1, T, 4]
    def dummy_embed(ids):
        return mx.zeros((1, len(ids), 4), dtype=mx.float32)

    model = types.SimpleNamespace(_vision_ctx=None)
    wrapped = iv.VisionAwareEmbed(dummy_embed, model)

    emb1 = mx.arange(1, 21, dtype=mx.float32).reshape(5, 4)  # span [3, 8)
    emb2 = mx.arange(101, 113, dtype=mx.float32).reshape(3, 4)  # span [10, 13)
    model._vision_ctx = {
        "encoded_spans": [(3, 8, emb1), (10, 13, emb2)],
        "chunk_pos0": 0,
    }

    # Chunk 0: pos [0, 6) -> text [0..3), emb1[0..3)
    h0 = wrapped([0] * 6)
    assert h0.shape == (1, 6, 4)
    assert np.allclose(np.array(h0[0, :3]), 0.0)
    assert np.allclose(np.array(h0[0, 3:6]), np.array(emb1[:3]))

    # Chunk 1: pos [6, 12) -> emb1[3..5), text [8..10), emb2[0..2)
    model._vision_ctx["chunk_pos0"] = 6
    h1 = wrapped([0] * 6)
    assert np.allclose(np.array(h1[0, 0:2]), np.array(emb1[3:5]))
    assert np.allclose(np.array(h1[0, 2:4]), 0.0)
    assert np.allclose(np.array(h1[0, 4:6]), np.array(emb2[0:2]))

    # Decode step (chunk_pos0 is None) -> pure text embedding
    model._vision_ctx["chunk_pos0"] = None
    hd = wrapped([42])
    assert np.allclose(np.array(hd), 0.0)
