"""Unit tests for safetensors vision_tower extractor and server CLI/normalization."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from inkling_mlx_vision import extract as ix
from inkling_mlx_vision import server as isrv


def _write_dummy_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    offset = 0
    chunks = []
    for name, arr in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        chunks.append(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for c in chunks:
            f.write(c)


def test_extract_from_local_preserves_exact_tensors(tmp_path: Path):
    src = tmp_path / "model-00001-of-00002.safetensors"
    out = tmp_path / "vision_tower.safetensors"
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 4)
    c = np.ones((5, 5), dtype=np.float32)
    _write_dummy_safetensors(
        src,
        {
            "llm.layer.0.weight": c,
            "vision_tower.backbone.ln_pre.weight": a,
            "vision_tower.proj_out.weight": b,
        },
    )

    info = ix.extract_from_local(src, out)
    assert info["tensors"] == 2
    assert info["bytes"] == a.nbytes + b.nbytes

    loaded = mx.load(str(out))
    assert set(loaded.keys()) == {"vision_tower.backbone.ln_pre.weight", "vision_tower.proj_out.weight"}
    assert np.allclose(np.array(loaded["vision_tower.backbone.ln_pre.weight"]), a)
    assert np.allclose(np.array(loaded["vision_tower.proj_out.weight"]), b)


def test_normalize_messages_user_and_tool_images():
    class DummyErr(Exception):
        def __init__(self, status, msg):
            super().__init__(msg)
            self.status = status

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "Screenshot",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,CCC"}},
                {"type": "text", "text": "Captured."},
            ],
        },
    ]
    norm, imgs = isrv.normalize_messages_with_vision(msgs, vision_enabled=True, request_error_cls=DummyErr)
    assert len(imgs) == 3
    assert norm[0]["content"] == [{"type": "text", "text": "Compare these:"}, {"type": "image"}, {"type": "image"}]
    assert norm[1]["content"] == [{"type": "image"}, {"type": "text", "text": "Captured."}]

    # Assistant images must be rejected cleanly
    with pytest.raises(DummyErr) as exc:
        isrv.normalize_messages_with_vision(
            [{"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}],
            vision_enabled=True,
            request_error_cls=DummyErr,
        )
    assert exc.value.status == 400


def test_parse_extra_args():
    cfg, rest = isrv.parse_extra_args(
        [
            "my-model",
            "--vision",
            "vt.safetensors",
            "--vision-max-pixels=1000000",
            "--allow-remote-images",
            "--metal-cache-mb",
            "512",
            "--no-loop-guard",
            "--port",
            "8080",
        ]
    )
    assert cfg.vision_path == Path("vt.safetensors")
    assert cfg.vision_max_pixels == 1000000
    assert cfg.allow_remote_images is True
    assert cfg.metal_cache_mb == 512
    assert cfg.loop_guard is False
    assert rest == ["my-model", "--port", "8080"]
