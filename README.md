# inkling-mlx-vision

**Native MLX Vision Tower + Agent-IDE hardened OpenAI server wrapper for [LookAtMyQuant](https://github.com/ShinMegamiBoson/lookatmyquant) `Inkling-2-Mini` on Apple Silicon.**

Adds full multimodal image input (`image_url` / base64 screenshots), zero-overhead lazy vision loading, multi-chat queueing, dynamic preamble prefix pinning, and long-context stability for 16–24+ GB Macs running coding agents (ZCode, Cline, Continue, Pi, OpenAI SDK).

---

## Key Features

- **Native Pure-MLX Vision Tower (`vision.py`)** — Implements Hugging Face `Inkling2ImageProcessor` + `Inkling2VisionModel` line-for-line in pure MLX + NumPy + Pillow (no PyTorch or torchvision required):
  - RGBA 8×8 checkerboard compositing, `-1.0` bottom-right padding to `40×40` patches, aspect-preserving downscaling for 4K/Retina screenshots (`1920×1080` kept 1:1).
  - 23-layer BF16 ViT with bilinear `32×32` position embedding interpolation, per-image CLS token, axial 2D `[w, h]` pairwise RoPE, LayerScale, exact erf-GELU, and `2×2` spatial merger (`4096 → 4096 → 2304`).
  - **0.00 GB RAM overhead on text turns:** `vision_tower.safetensors` (~637 MB) is memory-mapped on demand (~21 ms) and released immediately after encoding before LLM prefill starts.
  - Supports multiple images per message and images returned inside `role: "tool"` messages.
- **Zero-Bloat Vision Weight Extractor (`inkling-extract-vision`)** — Extracts the 422 BF16 `vision_tower.*` tensors (~637 MB) directly from Hugging Face using **HTTP Range requests** without downloading the full 16.9 GB checkpoint (or extracts in `<0.5s` from a local checkpoint).
- **Vision-Safe `SmartPromptCache` (`cache.py`)**
  - Injects deterministic image-hash surrogate IDs into cache keys: follow-up turns on the same screenshot skip ViT completely (`0 ms`), while two different images of identical resolution never collide.
  - **Dynamic preamble pinning:** automatically detects shared `system` + `tools` + leading `<system-reminder>` blocks (~24K tokens in agent IDEs) and pins a snapshot so new chats start in ~1–2s instead of ~60s; refreshes automatically when memory/MCP tools/date change.
  - High text `min_tokens` (default `2048`) prevents short auto-title requests from evicting active coding chats.
- **Agent-IDE Server Stability (`queue.py`, `parser.py`, `server.py`)**
  - **Fair FIFO queue** instead of `HTTP 503 model is busy` + **continuous SSE keepalive** (`: queued` / `: prefilling` / `: generating`) on any pause `>= 5s`, preventing client idle timeouts during long tool calls.
  - **24 GB Unified Memory safety:** `LAMQ_GLOBAL_KV=fp8`, bounded Metal buffer cache (`--metal-cache-mb 256`), chunked prefill (`--prefill-chunk 256`), and `64 MiB` request body limit for 128K contexts + base64 screenshots.
  - **Live token/state monitoring & loop guard:** logs parser state (`thinking`, `text`, `tool(name, buf=...)`, `discard_tool`) every 2s and cleanly stops verbatim 256-token repetition loops or stuck `discard_tool` runs.
  - **SSRF protection:** remote `http(s)://` image URLs are disabled by default (`--allow-remote-images` enables them with private/loopback/link-local IP blocking); `file://` URLs are only allowed on loopback hosts.

---

## Quickstart

### 1. Install

Requires macOS on Apple Silicon (M1/M2/M3/M4) and Python 3.11+.

```zsh
git clone https://github.com/qoder3/inkling-mlx-vision.git
cd inkling-mlx-vision
uv venv --python 3.12
source .venv/bin/activate
uv pip install "git+https://github.com/ShinMegamiBoson/lookatmyquant.git"
uv pip install -e .
```

### 2. Extract `vision_tower.safetensors` (~637 MB)

Download **only** the 637 MB vision weights from Hugging Face via HTTP Range requests (skips the 16.9 GB LLM shards):

```zsh
inkling-extract-vision -o vision_tower.safetensors
```

*(Already have `Inkling-2-Mini-oQ4e-mtp` locally? Extract in `<0.5s` offline:)*

```zsh
inkling-extract-vision --from-local /path/to/Inkling-2-Mini-oQ4e-mtp -o vision_tower.safetensors
```

### 3. Start the Server

```zsh
LAMQ_GLOBAL_KV=fp8 inkling-mlx-serve \
  shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB \
  --drafter shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter \
  --vision ./vision_tower.safetensors \
  --host 127.0.0.1 --port 8080 \
  --model-id inkling-2-mini \
  --context 131072 --max-tokens 16384 \
  --prompt-cache 3 --prefill-chunk 256 \
  --reasoning-effort medium --temp 0.6 --top-p 0.95
```

Or use the portable launcher script (supports `--restart` and env overrides):

```zsh
./scripts/start_server.sh --restart
```

---

## Benchmarks (Apple M4 Pro, 24 GB Unified Memory)

Measured with `Inkling-2-Mini-LookAtMyQuant-8.9GB` + 4-bit DSpark drafter + `LAMQ_GLOBAL_KV=fp8` + `--prefill-chunk 256` + `--context 131072`:

| Workload | Speed / Latency | Memory (Active / Peak) |
|---|---|---|
| Model + 4-bit drafter idle load | 5.3s load | 8.83 GB |
| Vision encoding (`480×320`, 96 tok) | 75–220 ms | 9.21 GB (0 GB retained after ViT) |
| Vision encoding (`1280×800`, 640 tok) | 571 ms | 9.27 GB / 10.87 GB peak |
| Vision encoding (`1920×1080`, 1,296 tok) | 1.15 s | +11.9 MB embeddings |
| Follow-up turn on cached image | **0 ms ViT**, 0.3s prefill | 9.27 GB |
| Prefill (`chunk=256`, 24K–32K ctx) | 260–330 tok/s | 10.8–12.9 GB peak |
| Decode (with 4-bit DSpark drafter) | **24–45 tok/s** | 11–12 GB |

### Memory Budget for 128K Context (`LAMQ_GLOBAL_KV=fp8`)

- Packed model + 4-bit drafter: **8.83 GB**
- Global KV (`fp8`) + drafter context: **~20 KiB / token** (`~0.65 GB` at 32K, `~1.3 GB` at 64K, `~2.6 GB` at 128K)
- Vision Tower (lazy mmap): **0.00 GB** during text turns / LLM prefill

---

## Client Configuration

### ZCode (`~/.zcode/v2/provider_config.json`)

To make ZCode send screenshots as `image_url` instead of stripping them, include `inputFormat.supportsImage: true` under `properties`:

```json
{
  "modelId": "inkling-2-mini",
  "config": {
    "enabled": true,
    "properties": {
      "contextWindow": 131072,
      "inputFormat": {
        "supportsImage": true
      }
    },
    "optionSpecs": {
      "maxOutputTokens": {
        "max": 16384
      }
    }
  },
  "providerId": "your-local-provider-id"
}
```

### OpenAI Python SDK / `curl`

Standard OpenAI Chat Completions multimodal format (`data:image/png;base64,...`):

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
b64 = base64.b64encode(open("screenshot.png", "rb").read()).decode()

resp = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "What error is shown in this terminal screenshot?"},
        ],
    }],
)
print(resp.choices[0].message.content)
```

---

## Wrapper CLI Flags

All standard `lamq serve` flags are supported alongside these wrapper options:

| Flag | Default | Description |
|---|---|---|
| `--vision PATH` | `./vision_tower.safetensors` (if present) | Path to extracted BF16 `vision_tower.safetensors` |
| `--no-vision` | `false` | Disable vision tower explicitly |
| `--vision-max-pixels INT` | `2457600` (`1920×1280`) | Aspect-preserving downscale threshold |
| `--vision-max-side INT` | `2560` | Maximum image width or height after downscale |
| `--allow-remote-images` | `false` | Allow fetching public `http(s)://` image URLs (private IPs blocked) |
| `--metal-cache-mb INT` | `256` | MLX Metal buffer cache cap in MiB |
| `--max-request-mb INT` | `64` | Maximum HTTP JSON request body size in MiB |
| `--cache-min-tokens INT` | `2048` | Minimum text prompt length to cache (short titles bypass cache) |
| `--pin-min-tokens INT` | `4096` | Minimum shared preamble length to pin in `SmartPromptCache` |
| `--no-loop-guard` | `false` | Disable verbatim repetition-loop & stuck `discard_tool` guards |
| `--log-file PATH` | `None` (stderr only) | Also append timestamped logs to a file |

---

## Development & Tests

Run the offline unit test suite (requires no model weights):

```zsh
uv pip install -e ".[dev]"
pytest
```

Run live multimodal E2E tests (including multi-image messages and `role: "tool"` images) against a running server:

```zsh
INKLING_LIVE_URL=http://127.0.0.1:8080 pytest tests/test_live_e2e.py -s
```

---

## Credits & License

- Built on top of [ShinMegamiBoson/lookatmyquant](https://github.com/ShinMegamiBoson/lookatmyquant) (MIT License).
- Vision architecture follows Hugging Face `transformers` `Inkling2VisionModel` / `Inkling2ImageProcessor` for Thinking Machines' `Inkling-2-Mini`.
- Released under the [MIT License](LICENSE). Model weights are **not** included and remain subject to their respective upstream licenses.
