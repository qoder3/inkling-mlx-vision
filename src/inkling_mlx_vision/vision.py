"""Pure-MLX Inkling-2 Vision Tower (ViT + 2x2 Merger) matching Hugging Face `transformers`.

Implements:
- `Inkling2ImageProcessor` equivalent: EXIF transpose, RGBA 8x8 (204/153) checkerboard composite,
  optional aspect-preserving max-pixel cap, bottom-right `-1.0` padding to 40x40 grid,
  rescale `1/255`, normalize `mean=0.5 / std=0.5`, patchify to `(rows*cols, 40, 40, 3)`.
- `Inkling2VisionModel` equivalent: 20x20 patch embedding, bilinear `32x32` position embedding
  interpolation in 2x2 spatial-merge-block order, per-image CLS token at RoPE `(0, 0)`, Pre-LN,
  23-layer ViT with axial 2D pairwise RoPE (`[freq_w, freq_h]` order), LayerScale, exact erf GELU,
  and 2x2 spatial merger projecting `4096 -> 4096 -> 2304`.
- `VisionAwareEmbed`: splices precomputed vision embeddings into chunked prefill hidden states
  without modifying `Inkling2LoopPackedMLX` internals.

Weights are memory-mapped lazily on demand (~21 ms for 637 MB BF16) and released immediately after
encoding so text-only requests carry 0.00 GB vision RAM overhead.
"""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import math
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image, ImageOps

PATCH_SIZE = 20
MERGE_SIZE = 2
BLOCK_SIZE = PATCH_SIZE * MERGE_SIZE  # 40 px per LLM image token
POSEMB_GRID_SIZE = 32
HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64
NUM_LAYERS = 23
LAYER_NORM_EPS = 1e-5
ROPE_THETA = 10000.0
TEXT_HIDDEN_SIZE = 2304

# Keeps 1920x1080 screenshots 100% unscaled while capping extreme 4K/5K Retina screenshots
# so ViT latency and token counts stay fast on 16-24 GB Apple Silicon Macs.
DEFAULT_MAX_PIXELS = 1920 * 1280  # ~1,536 image tokens max
DEFAULT_MAX_SIDE = 2560


@dataclass
class PreprocessedImage:
    pixel_values: np.ndarray  # float32 [rows * cols, 40, 40, 3]
    rows: int
    cols: int
    img_hash: str  # 32-char hex blake2b digest
    orig_size: tuple[int, int]  # (W, H)
    proc_size: tuple[int, int]  # (W, H)

    @property
    def num_tokens(self) -> int:
        return self.rows * self.cols


def _validate_remote_host(url: str) -> None:
    """Reject non-public IPs (loopback, link-local, RFC1918, multicast) to prevent SSRF."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError("remote image URL is missing a hostname")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError(f"failed to resolve remote image host {host!r}: {e}") from e
    if not infos:
        raise ValueError(f"could not resolve remote image host {host!r}")
    for _, _, _, _, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError(f"remote image URL resolves to non-public IP address ({ip})")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        resolved = urllib.parse.urljoin(req.full_url, newurl)
        if not resolved.startswith(("http://", "https://")):
            raise ValueError("redirect to non-HTTP(S) scheme is not allowed")
        _validate_remote_host(resolved)
        return super().redirect_request(req, fp, code, msg, headers, resolved)


def extract_image_spec(part: dict[str, Any]) -> tuple[str, str]:
    """Return `(kind, value)` from an OpenAI, Anthropic, or Responses-API image content part.

    `kind` is `"url"` (`data:`, `http(s):`, or `file://`) or `"base64"`.
    """
    ptype = part.get("type")
    if ptype in ("image_url", "input_image"):
        iu = part.get("image_url")
        if isinstance(iu, dict):
            url = iu.get("url")
            if isinstance(url, str) and url:
                return "url", url
        elif isinstance(iu, str) and iu:
            return "url", iu
        if isinstance(part.get("url"), str) and part["url"]:
            return "url", part["url"]
    if ptype == "image":
        src = part.get("source")
        if isinstance(src, dict):
            if src.get("type") == "base64" and isinstance(src.get("data"), str):
                return "base64", src["data"]
            if src.get("type") == "url" and isinstance(src.get("url"), str):
                return "url", src["url"]
        for key in ("image", "url", "image_url"):
            val = part.get(key)
            if isinstance(val, dict) and isinstance(val.get("url"), str) and val["url"]:
                return "url", val["url"]
            if isinstance(val, str) and val:
                return "url", val
    raise ValueError("image content part is missing a valid image_url / url / base64 source")


def load_image_bytes(
    part: dict[str, Any],
    *,
    allow_file: bool = False,
    allow_remote: bool = False,
    max_bytes: int = 32 << 20,
    timeout: float = 15.0,
) -> tuple[bytes, str]:
    """Decode or fetch raw image bytes from a message content part."""
    kind, value = extract_image_spec(part)
    value = value.strip()

    if kind == "base64":
        raw = base64.b64decode(value)
    elif value.startswith("data:"):
        comma = value.find(",")
        if comma < 0:
            raise ValueError("invalid data URI for image")
        header, payload = value[:comma], value[comma + 1 :]
        if ";base64" in header.lower():
            raw = base64.b64decode(payload)
        else:
            raw = urllib.parse.unquote_to_bytes(payload)
    elif value.startswith(("http://", "https://")):
        if not allow_remote:
            raise ValueError("remote http(s) image URLs are disabled (pass --allow-remote-images to enable)")
        _validate_remote_host(value)
        opener = urllib.request.build_opener(_SafeRedirectHandler())
        req = urllib.request.Request(value, headers={"User-Agent": "inkling-mlx-vision/0.1"})
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read(max_bytes + 1)
    elif value.startswith("file://"):
        if not allow_file:
            raise ValueError("file:// image URLs are only allowed on loopback hosts")
        parsed = urllib.parse.urlparse(value)
        fpath = Path(urllib.parse.unquote(parsed.path))
        if not fpath.is_file():
            raise ValueError(f"local image file not found: {fpath}")
        if fpath.stat().st_size > max_bytes:
            raise ValueError("image exceeds maximum size")
        raw = fpath.read_bytes()
    else:
        raise ValueError("unsupported image URL scheme (expected data:, or enabled http(s): / file://)")

    if not raw:
        raise ValueError("empty image payload")
    if len(raw) > max_bytes:
        raise ValueError(f"image exceeds maximum allowed size ({max_bytes // (1 << 20)} MiB)")

    digest = hashlib.blake2b(raw, digest_size=16).hexdigest()
    return raw, digest


def preprocess_image(
    raw_bytes: bytes,
    img_hash: str | None = None,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
) -> PreprocessedImage:
    """Preprocess raw image bytes into Inkling-2 `(rows*cols, 40, 40, 3)` float32 patches."""
    if img_hash is None:
        img_hash = hashlib.blake2b(raw_bytes, digest_size=16).hexdigest()

    with Image.open(io.BytesIO(raw_bytes)) as opened:
        img = ImageOps.exif_transpose(opened)
        orig_w, orig_h = img.size

        # Match Inkling2ImageProcessor: composite transparent images onto an 8x8 (204/153) checkerboard.
        has_alpha = (
            getattr(img, "has_transparency_data", False)
            or img.mode in ("RGBA", "LA", "PA")
            or ("transparency" in img.info)
        )
        if has_alpha:
            rgba = np.asarray(img.convert("RGBA"), dtype=np.int32)
            h, w = rgba.shape[:2]
            r_idx = np.arange(h, dtype=np.int32)[:, None] // 8
            c_idx = np.arange(w, dtype=np.int32)[None, :] // 8
            bg = np.where((r_idx + c_idx) % 2 == 0, 204, 153)[:, :, None]
            alpha = rgba[:, :, 3:4]
            rgb = ((rgba[:, :, :3] * alpha + bg * (255 - alpha) + 127) // 255).astype(np.uint8)
        else:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("image has zero width or height")

    # Cap extreme resolutions safely while preserving aspect ratio.
    if (max_pixels > 0 and h * w > max_pixels) or (max_side > 0 and max(h, w) > max_side):
        scale_px = math.sqrt(max_pixels / (h * w)) if (max_pixels > 0 and h * w > max_pixels) else 1.0
        scale_side = (max_side / max(h, w)) if (max_side > 0 and max(h, w) > max_side) else 1.0
        scale = min(scale_px, scale_side)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        rgb = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
        h, w = new_h, new_w

    rows = math.ceil(h / BLOCK_SIZE)
    cols = math.ceil(w / BLOCK_SIZE)
    pad_h = rows * BLOCK_SIZE - h
    pad_w = cols * BLOCK_SIZE - w

    arr = rgb.astype(np.float32)
    if pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=-1.0)

    # Exact Inkling2ImageProcessor rescale + normalize: (x / 255 - 0.5) / 0.5
    arr = (arr * (1.0 / 255.0) - 0.5) / 0.5

    patches = (
        arr.reshape(rows, BLOCK_SIZE, cols, BLOCK_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(rows * cols, BLOCK_SIZE, BLOCK_SIZE, 3)
    )
    return PreprocessedImage(
        pixel_values=np.ascontiguousarray(patches, dtype=np.float32),
        rows=rows,
        cols=cols,
        img_hash=img_hash,
        orig_size=(orig_w, orig_h),
        proc_size=(w, h),
    )


def _bilinear_axis_taps_weights(index: np.ndarray, size: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact match to `transformers.vision_utils._interpolation_axis_taps_weights` (bilinear, align_corners=False)."""
    idx_f = index.astype(np.float32)
    src = (idx_f + 0.5) * float(side) / float(size) - 0.5
    floor = np.floor(src)
    offsets = np.array([0, 1], dtype=np.int32)
    raw_taps = floor.astype(np.int32)[:, None] + offsets[None, :]
    taps = np.clip(raw_taps, 0, side - 1)
    distance = np.abs(src[:, None] - floor[:, None] - offsets[None, :].astype(np.float32))
    weights = np.maximum(1.0 - distance, 0.0).astype(np.float32)
    return taps, weights


def _vision_grid_coords(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Return `(row, col)` native patch coordinates in 2x2 spatial-merge-block order."""
    h_p = rows * MERGE_SIZE
    w_p = cols * MERGE_SIZE
    h_grid, w_grid = np.meshgrid(
        np.arange(h_p, dtype=np.int32),
        np.arange(w_p, dtype=np.int32),
        indexing="ij",
    )
    row = h_grid.reshape(rows, MERGE_SIZE, cols, MERGE_SIZE).transpose(0, 2, 1, 3).reshape(-1)
    col = w_grid.reshape(rows, MERGE_SIZE, cols, MERGE_SIZE).transpose(0, 2, 1, 3).reshape(-1)
    return row, col


def _pos_interp_indices_and_weights(rows: int, cols: int) -> tuple[mx.array, mx.array]:
    """Bilinear interpolation indices `[N_p, 4]` and weights `[N_p, 4]` into the 32x32 learned pos table."""
    h_p = rows * MERGE_SIZE
    w_p = cols * MERGE_SIZE
    row, col = _vision_grid_coords(rows, cols)
    h_taps, h_w = _bilinear_axis_taps_weights(row, h_p, POSEMB_GRID_SIZE)
    w_taps, w_w = _bilinear_axis_taps_weights(col, w_p, POSEMB_GRID_SIZE)
    indices = (h_taps[:, :, None] * POSEMB_GRID_SIZE + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return mx.array(indices, dtype=mx.int32), mx.array(weights, dtype=mx.float32)


_INV_FREQ = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM // 2, 2, dtype=np.float32) / float(HEAD_DIM // 2)))


def _axial_rope_cos_sin(rows: int, cols: int) -> tuple[mx.array, mx.array]:
    """Compute `(cos, sin)` of shape `[1 + 4*rows*cols, 1, 64]` matching `Inkling2VisionRotaryEmbedding`.

    CLS token sits at `(0, 0)`, patches are 1-indexed `(row+1, col+1)`,
    and frequencies are ordered `[freq_w, freq_h]` with each entry repeated twice for pairwise rotation.
    """
    row, col = _vision_grid_coords(rows, cols)
    h_all = np.concatenate([[0], row + 1]).astype(np.float32)
    w_all = np.concatenate([[0], col + 1]).astype(np.float32)
    freqs_w = w_all[:, None] * _INV_FREQ[None, :]
    freqs_h = h_all[:, None] * _INV_FREQ[None, :]
    freqs = np.concatenate([freqs_w, freqs_h], axis=-1)  # [1 + N_p, 32]
    cos = np.repeat(np.cos(freqs), 2, axis=-1).astype(np.float32)  # [1 + N_p, 64]
    sin = np.repeat(np.sin(freqs), 2, axis=-1).astype(np.float32)  # [1 + N_p, 64]
    return mx.array(cos)[:, None, :], mx.array(sin)[:, None, :]


def _rotate_pairwise(x: mx.array) -> mx.array:
    """Rotate each interleaved pair `(x1, x2)` -> `(-x2, x1)`."""
    x_pair = x.reshape(*x.shape[:-1], -1, 2)
    return mx.stack([-x_pair[..., 1], x_pair[..., 0]], axis=-1).reshape(x.shape)


def _apply_rope_vision(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    orig_dtype = q.dtype
    qf = q.astype(mx.float32)
    kf = k.astype(mx.float32)
    q_out = (qf * cos + _rotate_pairwise(qf) * sin).astype(orig_dtype)
    k_out = (kf * cos + _rotate_pairwise(kf) * sin).astype(orig_dtype)
    return q_out, k_out


def encode_single_image(img: PreprocessedImage, w: dict[str, mx.array]) -> mx.array:
    """Encode one `PreprocessedImage` using loaded BF16 vision weights -> float32 `[rows*cols, 2304]`."""
    rows, cols = img.rows, img.cols
    n_tok = rows * cols
    n_patches = n_tok * (MERGE_SIZE * MERGE_SIZE)
    seq_len = 1 + n_patches

    patch_w = w["vision_tower.backbone.embeddings.patch_embedding.weight"]  # [1024, 1200] bf16
    dtype = patch_w.dtype

    pv = mx.array(img.pixel_values, dtype=dtype)  # [n_tok, 40, 40, 3]
    patches = (
        pv.reshape(n_tok, MERGE_SIZE, PATCH_SIZE, MERGE_SIZE, PATCH_SIZE, 3)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(n_patches, PATCH_SIZE * PATCH_SIZE * 3)
    )
    patch_emb = patches @ patch_w.T  # [n_patches, 1024]

    interp_idx, interp_w = _pos_interp_indices_and_weights(rows, cols)
    pos_table = w["vision_tower.backbone.embeddings.position_embedding"]  # [1024, 1024]
    pos_emb = mx.sum(pos_table[interp_idx].astype(mx.float32) * interp_w[:, :, None], axis=1)
    patch_emb = patch_emb + pos_emb.astype(dtype)

    cls_emb = (
        w["vision_tower.backbone.embeddings.class_embedding"]
        + w["vision_tower.backbone.embeddings.cls_position_embedding"]
    ).astype(dtype)[None, :]  # [1, 1024]
    h = mx.concatenate([cls_emb, patch_emb], axis=0)  # [1 + n_patches, 1024]

    h = mx.fast.layer_norm(
        h,
        w["vision_tower.backbone.ln_pre.weight"],
        w["vision_tower.backbone.ln_pre.bias"],
        LAYER_NORM_EPS,
    )

    cos, sin = _axial_rope_cos_sin(rows, cols)
    scale = HEAD_DIM**-0.5

    for i in range(NUM_LAYERS):
        p = f"vision_tower.backbone.layers.{i}"
        residual = h
        x = mx.fast.layer_norm(h, w[f"{p}.ln_1.weight"], w[f"{p}.ln_1.bias"], LAYER_NORM_EPS)

        q = (x @ w[f"{p}.self_attn.q_proj.weight"].T + w[f"{p}.self_attn.q_proj.bias"]).reshape(
            seq_len, NUM_HEADS, HEAD_DIM
        )
        k = (x @ w[f"{p}.self_attn.k_proj.weight"].T + w[f"{p}.self_attn.k_proj.bias"]).reshape(
            seq_len, NUM_HEADS, HEAD_DIM
        )
        v = (x @ w[f"{p}.self_attn.v_proj.weight"].T + w[f"{p}.self_attn.v_proj.bias"]).reshape(
            seq_len, NUM_HEADS, HEAD_DIM
        )

        q, k = _apply_rope_vision(q, k, cos, sin)

        q_sdpa = q.transpose(1, 0, 2)[None, :, :, :]
        k_sdpa = k.transpose(1, 0, 2)[None, :, :, :]
        v_sdpa = v.transpose(1, 0, 2)[None, :, :, :]
        attn_out = mx.fast.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=scale)
        attn_out = attn_out[0].transpose(1, 0, 2).reshape(seq_len, HIDDEN_SIZE)
        attn_out = attn_out @ w[f"{p}.self_attn.out_proj.weight"].T + w[f"{p}.self_attn.out_proj.bias"]

        h = residual + w[f"{p}.ls_1.gamma"] * attn_out

        residual = h
        x2 = mx.fast.layer_norm(h, w[f"{p}.ln_2.weight"], w[f"{p}.ln_2.bias"], LAYER_NORM_EPS)
        mlp = x2 @ w[f"{p}.mlp.c_fc.weight"].T + w[f"{p}.mlp.c_fc.bias"]
        mlp = nn.gelu(mlp)
        mlp = mlp @ w[f"{p}.mlp.c_proj.weight"].T + w[f"{p}.mlp.c_proj.bias"]

        h = residual + w[f"{p}.ls_2.gamma"] * mlp
        if (i + 1) % 4 == 0:
            mx.async_eval(h)

    patch_states = h[1:].reshape(n_tok, MERGE_SIZE * MERGE_SIZE * HIDDEN_SIZE)  # [n_tok, 4096]
    merged = patch_states @ w["vision_tower.proj_in.weight"].T  # [n_tok, 4096]
    merged = nn.gelu(merged)
    out = merged @ w["vision_tower.proj_out.weight"].T  # [n_tok, 2304]
    out_f32 = out.astype(mx.float32)
    mx.eval(out_f32)
    return out_f32


def encode_images(images: list[PreprocessedImage], weights_path: str | Path) -> list[mx.array]:
    """Load vision weights lazily via mmap, encode all images, and immediately release weights."""
    if not images:
        return []
    weights = mx.load(str(weights_path))
    try:
        return [encode_single_image(img, weights) for img in images]
    finally:
        del weights
        mx.clear_cache()


class VisionAwareEmbed:
    """Wraps `PackedEmbed` to splice precomputed vision embeddings at image positions during chunked prefill."""

    def __init__(self, orig_embed, model):
        self._orig = orig_embed
        self._m = model

    def __getattr__(self, name: str):
        return getattr(self._orig, name)

    def __call__(self, input_ids):
        h = self._orig(input_ids)  # [1, T, 2304] float32
        vctx = getattr(self._m, "_vision_ctx", None)
        if vctx is None:
            return h
        pos0 = vctx.get("chunk_pos0")
        spans = vctx.get("encoded_spans")
        if pos0 is None or not spans:
            return h

        T = int(h.shape[1])
        pos1 = pos0 + T
        overlaps: list[tuple[int, int, mx.array]] = []
        for s, e, emb in spans:
            ov_s = max(pos0, s)
            ov_e = min(pos1, e)
            if ov_s < ov_e:
                overlaps.append((ov_s - pos0, ov_e - pos0, emb[ov_s - s : ov_e - s]))
        if not overlaps:
            return h

        overlaps.sort(key=lambda x: x[0])
        pieces: list[mx.array] = []
        cur = 0
        for loc_s, loc_e, emb_slice in overlaps:
            if loc_s > cur:
                pieces.append(h[:, cur:loc_s, :])
            pieces.append(emb_slice.astype(h.dtype)[None, :, :])
            cur = loc_e
        if cur < T:
            pieces.append(h[:, cur:T, :])
        if len(pieces) == 1:
            return pieces[0]
        return mx.concatenate(pieces, axis=1)
