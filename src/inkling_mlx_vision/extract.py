"""Zero-bloat extractor for Inkling-2 BF16 `vision_tower.safetensors` (~636.6 MB).

Supports:
1. Direct HTTP Range extraction from Hugging Face (downloads ONLY the ~637 MB `vision_tower.*`
   byte spans instead of the full 16.9 GB checkpoint).
2. Local extraction from an existing checkpoint directory or `.safetensors` file (<0.5s, zero
   PyTorch/MLX dependency).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

DEFAULT_HF_REPO = "mlx-community/Inkling-2-Mini-oQ4e-mtp"
DEFAULT_PREFIX = "vision_tower."
DEFAULT_OUTPUT = "vision_tower.safetensors"
_USER_AGENT = "inkling-mlx-vision/0.1"


def _hf_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"User-Agent": _USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    if extra:
        h.update(extra)
    return h


def _http_get(url: str, *, byte_range: tuple[int, int] | None = None, timeout: float = 60.0, retries: int = 3) -> bytes:
    extra = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range is not None else None
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_hf_headers(extra))
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if byte_range is not None:
                expected = byte_range[1] - byte_range[0] + 1
                if len(data) != expected:
                    raise IOError(f"short Range read: expected {expected} bytes, got {len(data)}")
            return data
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"HTTP fetch failed for {url}: {last_err}") from last_err


def _read_local_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        hdr_len = struct.unpack("<Q", raw_len)[0]
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
    return hdr, 8 + hdr_len


def _read_remote_header(url: str) -> tuple[dict[str, Any], int]:
    hdr_len = struct.unpack("<Q", _http_get(url, byte_range=(0, 7)))[0]
    hdr = json.loads(_http_get(url, byte_range=(8, 8 + hdr_len - 1)).decode("utf-8"))
    return hdr, 8 + hdr_len


def _coalesce_spans(items: list[tuple[str, dict[str, Any]]], merge_gap: int = 256 * 1024) -> list[tuple[int, int, list[tuple[str, int, int]]]]:
    """Group `(name, meta)` entries sorted by source start offset into contiguous/near-contiguous windows."""
    windows: list[tuple[int, int, list[tuple[str, int, int]]]] = []
    for name, meta in items:
        s, e = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        if windows and s - windows[-1][1] <= merge_gap:
            ws, we, group = windows[-1]
            group.append((name, s, e))
            windows[-1] = (ws, max(we, e), group)
        else:
            windows.append((s, e, [(name, s, e)]))
    return windows


def extract_from_local(
    source: str | Path,
    output: str | Path,
    *,
    prefix: str = DEFAULT_PREFIX,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Extract all tensors starting with `prefix` from a local checkpoint directory or `.safetensors` file."""
    src_path = Path(source).expanduser()
    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        shard_files = [src_path]
    elif src_path.is_dir():
        idx_file = src_path / "model.safetensors.index.json"
        if idx_file.is_file():
            weight_map = json.loads(idx_file.read_text(encoding="utf-8")).get("weight_map", {})
            shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
            shard_files = [src_path / s for s in shards]
        else:
            shard_files = sorted(src_path.glob("*.safetensors"))
    else:
        raise FileNotFoundError(f"source path does not exist: {src_path}")

    plan: list[tuple[Path, int, list[tuple[str, dict[str, Any]]]]] = []
    out_header: dict[str, Any] = {}
    out_offset = 0

    for sf in shard_files:
        hdr, data_base = _read_local_header(sf)
        matched = [
            (k, v)
            for k, v in hdr.items()
            if k != "__metadata__" and k.startswith(prefix) and isinstance(v, dict)
        ]
        if not matched:
            continue
        matched.sort(key=lambda kv: int(kv[1]["data_offsets"][0]))
        plan.append((sf, data_base, matched))
        for k, meta in matched:
            s, e = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            nbytes = e - s
            out_header[k] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [out_offset, out_offset + nbytes],
            }
            out_offset += nbytes

    if not out_header:
        raise ValueError(f"no tensors matching prefix {prefix!r} found in {src_path}")

    hdr_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    written = 0
    try:
        with tmp_path.open("wb") as out_f:
            out_f.write(struct.pack("<Q", len(hdr_bytes)))
            out_f.write(hdr_bytes)
            for sf, data_base, matched in plan:
                with sf.open("rb") as in_f:
                    for _, meta in matched:
                        s, e = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
                        rem = e - s
                        in_f.seek(data_base + s)
                        while rem > 0:
                            chunk = in_f.read(min(rem, 4 << 20))
                            if not chunk:
                                raise EOFError(f"unexpected EOF reading {sf}")
                            out_f.write(chunk)
                            rem -= len(chunk)
                            written += len(chunk)
                        if progress_cb is not None:
                            progress_cb(written, out_offset)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return {
        "tensors": len(out_header),
        "bytes": out_offset,
        "output": str(out_path),
    }


def extract_from_hf(
    repo_id: str = DEFAULT_HF_REPO,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    revision: str = "main",
    prefix: str = DEFAULT_PREFIX,
    merge_gap: int = 256 * 1024,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Download ONLY `prefix` tensors from a Hugging Face safetensors repo using HTTP Range requests."""
    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"https://huggingface.co/{repo_id}/resolve/{revision}"

    # Discover shard(s) containing prefix tensors.
    try:
        idx_data = json.loads(_http_get(f"{base_url}/model.safetensors.index.json").decode("utf-8"))
        weight_map = idx_data.get("weight_map", {})
        shard_names = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    except Exception:
        shard_names = ["model.safetensors"]

    if not shard_names:
        raise ValueError(f"no shards with prefix {prefix!r} found in {repo_id}")

    plan: list[tuple[str, int, list[tuple[str, dict[str, Any]]]]] = []
    out_header: dict[str, Any] = {}
    out_offset = 0

    for shard in shard_names:
        url = f"{base_url}/{shard}"
        hdr, data_base = _read_remote_header(url)
        matched = [
            (k, v)
            for k, v in hdr.items()
            if k != "__metadata__" and k.startswith(prefix) and isinstance(v, dict)
        ]
        if not matched:
            continue
        matched.sort(key=lambda kv: int(kv[1]["data_offsets"][0]))
        plan.append((url, data_base, matched))
        for k, meta in matched:
            s, e = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
            nbytes = e - s
            out_header[k] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [out_offset, out_offset + nbytes],
            }
            out_offset += nbytes

    if not out_header:
        raise ValueError(f"no tensors matching prefix {prefix!r} found in {repo_id}")

    hdr_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    written = 0
    downloaded = 0
    try:
        with tmp_path.open("wb") as out_f:
            out_f.write(struct.pack("<Q", len(hdr_bytes)))
            out_f.write(hdr_bytes)
            for url, data_base, matched in plan:
                windows = _coalesce_spans(matched, merge_gap=merge_gap)
                for ws, we, group in windows:
                    raw = _http_get(url, byte_range=(data_base + ws, data_base + we - 1))
                    downloaded += len(raw)
                    for _, s, e in group:
                        slice_bytes = raw[s - ws : e - ws]
                        out_f.write(slice_bytes)
                        written += len(slice_bytes)
                    if progress_cb is not None:
                        progress_cb(written, out_offset)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return {
        "tensors": len(out_header),
        "bytes": out_offset,
        "downloaded_bytes": downloaded,
        "output": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="inkling-extract-vision",
        description="Extract BF16 Inkling-2 vision_tower.safetensors (~637 MB) via HF HTTP Range or local checkpoint.",
    )
    p.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output safetensors path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--from-local",
        default=None,
        help="Local checkpoint directory or .safetensors file (skips network download)",
    )
    p.add_argument(
        "--repo",
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face repo ID for HTTP Range extraction (default: {DEFAULT_HF_REPO})",
    )
    p.add_argument("--revision", default="main", help="Hugging Face revision (default: main)")
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Tensor key prefix (default: {DEFAULT_PREFIX})")
    args = p.parse_args(argv)

    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        pct = 100.0 * done / max(total, 1)
        sys.stderr.write(f"\rExtracting {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
        sys.stderr.flush()

    if args.from_local:
        info = extract_from_local(args.from_local, args.output, prefix=args.prefix, progress_cb=progress)
    else:
        print(f"Fetching {args.prefix}* headers from {args.repo} (HTTP Range, no full 17 GB download)...", file=sys.stderr)
        info = extract_from_hf(args.repo, args.output, revision=args.revision, prefix=args.prefix, progress_cb=progress)

    dt = time.perf_counter() - t0
    sys.stderr.write("\n")
    print(
        f"Saved {info['output']}: {info['tensors']} tensors, {info['bytes'] / 1e6:.2f} MB in {dt:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
