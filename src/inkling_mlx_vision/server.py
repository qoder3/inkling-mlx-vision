"""CLI and runtime integration entrypoint (`inkling-mlx-serve`) for LookAtMyQuant Inkling-2-Mini."""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import accel as _accel_mod
from . import cache as _cache_mod
from . import parser as _parser_mod
from . import queue as _queue_mod
from . import vision as _vision


@dataclass
class ServerConfig:
    vision_path: Path | None = None
    vision_max_pixels: int = _vision.DEFAULT_MAX_PIXELS
    vision_max_side: int = _vision.DEFAULT_MAX_SIDE
    allow_remote_images: bool = False
    metal_cache_mb: int = 256
    max_request_mb: int = 64
    cache_min_tokens: int = 2048
    pin_min_tokens: int = 4096
    loop_guard: bool = True
    metal_accel: bool = True
    keep_wired: bool = True
    max_reasoning_effort: str | None = None
    log_file: Path | None = None


class _Logger:
    def __init__(self, log_file: Path | None = None):
        self._lock = threading.Lock()
        self._fp = log_file.open("a", encoding="utf-8", buffering=1) if log_file is not None else None

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        with self._lock:
            sys.stderr.write(line)
            sys.stderr.flush()
            if self._fp is not None:
                self._fp.write(line)
                self._fp.flush()


def normalize_messages_with_vision(
    msgs: Any,
    *,
    vision_enabled: bool,
    request_error_cls: type[Exception],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert OpenAI messages for Inkling's chat template, preserving image parts in appearance order."""
    if not isinstance(msgs, list) or not msgs:
        raise request_error_cls(400, "messages must be a non-empty list")
    out: list[dict[str, Any]] = []
    images_raw: list[dict[str, Any]] = []

    for m in msgs:
        if not isinstance(m, dict):
            raise request_error_cls(400, "each message must be an object")
        role = "system" if m.get("role") == "developer" else m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise request_error_cls(400, f"unsupported role: {m.get('role')!r}")
        content = m.get("content")
        if isinstance(content, list):
            has_img = any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image")
                for p in content
            )
            if not has_img:
                parts: list[str] = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict) and p.get("type") in (None, "text", "input_text"):
                        part = p.get("text")
                        if not isinstance(part, str):
                            raise request_error_cls(400, "text content parts require a string")
                        parts.append(part)
                content = "".join(parts)
            elif not vision_enabled:
                parts = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict) and p.get("type") in (None, "text", "input_text"):
                        part = p.get("text")
                        if not isinstance(part, str):
                            raise request_error_cls(400, "text content parts require a string")
                        parts.append(part)
                    elif isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"):
                        parts.append("[image omitted: vision weights not loaded]")
                content = "".join(parts)
            else:
                if role not in ("user", "tool"):
                    raise request_error_cls(400, "image content is only supported for user and tool messages")
                mm_parts: list[dict[str, Any]] = []
                text_buf: list[str] = []

                def flush_text() -> None:
                    if text_buf:
                        joined = "".join(text_buf)
                        text_buf.clear()
                        if joined:
                            mm_parts.append({"type": "text", "text": joined})

                for p in content:
                    if isinstance(p, str):
                        text_buf.append(p)
                    elif isinstance(p, dict) and p.get("type") in (None, "text", "input_text"):
                        part = p.get("text")
                        if not isinstance(part, str):
                            raise request_error_cls(400, "text content parts require a string")
                        text_buf.append(part)
                    elif isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"):
                        flush_text()
                        mm_parts.append({"type": "image"})
                        images_raw.append(p)
                    else:
                        bad = p.get("type") if isinstance(p, dict) else type(p).__name__
                        raise request_error_cls(400, f"unsupported content part type: {bad}")
                flush_text()
                content = mm_parts
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        nm: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            rc = m.get("reasoning_content") or m.get("reasoning") or m.get("reasoning_text")
            if isinstance(rc, str) and rc.strip():
                nm["reasoning_content"] = rc
            calls = []
            raw_calls = m.get("tool_calls")
            if raw_calls is not None and not isinstance(raw_calls, list):
                raise request_error_cls(400, "tool_calls must be an array")
            for tc in raw_calls or []:
                if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                    raise request_error_cls(400, "each tool call requires a function object")
                fn = tc["function"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except (ValueError, RecursionError):
                        args = {"input": args}
                if not isinstance(args, dict):
                    args = {"input": args}
                calls.append(
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {"name": fn.get("name", ""), "arguments": args},
                    }
                )
            if calls:
                nm["tool_calls"] = calls
        elif role == "tool":
            if m.get("tool_call_id"):
                nm["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                nm["name"] = m["name"]
        out.append(nm)
    return out, images_raw


def parse_extra_args(argv: list[str]) -> tuple[ServerConfig, list[str]]:
    """Extract wrapper-specific CLI flags before forwarding standard `lamq serve` flags."""
    default_vision = Path("vision_tower.safetensors")
    cfg = ServerConfig(vision_path=default_vision if default_vision.is_file() else None)
    rest: list[str] = []
    i = 0

    def take_val(flag: str) -> str:
        nonlocal i
        if "=" in argv[i]:
            val = argv[i].split("=", 1)[1]
            i += 1
            return val
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} requires a value")
        val = argv[i + 1]
        i += 2
        return val

    while i < len(argv):
        a = argv[i]
        if a == "--no-vision":
            cfg.vision_path = None
            i += 1
        elif a == "--vision" or a.startswith("--vision="):
            cfg.vision_path = Path(take_val("--vision")).expanduser()
        elif a == "--vision-max-pixels" or a.startswith("--vision-max-pixels="):
            cfg.vision_max_pixels = int(take_val("--vision-max-pixels"))
        elif a == "--vision-max-side" or a.startswith("--vision-max-side="):
            cfg.vision_max_side = int(take_val("--vision-max-side"))
        elif a == "--allow-remote-images":
            cfg.allow_remote_images = True
            i += 1
        elif a == "--metal-cache-mb" or a.startswith("--metal-cache-mb="):
            cfg.metal_cache_mb = int(take_val("--metal-cache-mb"))
        elif a == "--max-request-mb" or a.startswith("--max-request-mb="):
            cfg.max_request_mb = int(take_val("--max-request-mb"))
        elif a == "--cache-min-tokens" or a.startswith("--cache-min-tokens="):
            cfg.cache_min_tokens = int(take_val("--cache-min-tokens"))
        elif a == "--pin-min-tokens" or a.startswith("--pin-min-tokens="):
            cfg.pin_min_tokens = int(take_val("--pin-min-tokens"))
        elif a == "--no-loop-guard":
            cfg.loop_guard = False
            i += 1
        elif a == "--no-metal-accel":
            cfg.metal_accel = False
            i += 1
        elif a == "--no-keep-wired":
            cfg.keep_wired = False
            i += 1
        elif a == "--max-reasoning-effort" or a.startswith("--max-reasoning-effort="):
            cfg.max_reasoning_effort = take_val("--max-reasoning-effort").strip().lower() or None
        elif a == "--log-file" or a.startswith("--log-file="):
            cfg.log_file = Path(take_val("--log-file")).expanduser()
        else:
            rest.append(a)
            i += 1
    return cfg, rest


def install_patches(cfg: ServerConfig) -> None:
    """Install Vision, SmartPromptCache, FairQueue, keepalive, and monitoring hooks onto `runtime_mlx`."""
    os.environ.setdefault("LAMQ_GLOBAL_KV", "fp8")

    import mlx.core as mx
    from runtime_mlx import inkling as _inkling
    from runtime_mlx import server as _server
    from runtime_mlx.inkling import generate as _gen

    if cfg.metal_cache_mb > 0:
        mx.set_cache_limit(cfg.metal_cache_mb * 1024 * 1024)

    _server.MAX_REQUEST_BYTES = cfg.max_request_mb << 20
    _server.MAX_CONNECTIONS = 16
    _server.REQUEST_TIMEOUT = 30

    log_fn = _Logger(cfg.log_file)
    _server.log = log_fn

    if cfg.metal_accel:
        _accel_mod.install_metal_accelerations(log_fn=log_fn)

    _cache_mod.SmartPromptCache.default_min_tokens = cfg.cache_min_tokens
    _cache_mod.SmartPromptCache.pin_min_tokens = cfg.pin_min_tokens
    _cache_mod.SmartPromptCache.log_fn = staticmethod(log_fn)

    _parser_mod.MonitoredOutputParser.loop_guard_enabled = cfg.loop_guard
    _parser_mod.MonitoredOutputParser.log_fn = staticmethod(log_fn)
    _server.OutputParser = _parser_mod.MonitoredOutputParser

    def _instrumented_prefill(
        m,
        prompt_ids,
        cache,
        *,
        ordered: bool,
        prefill_chunk,
        tap_visits=(),
        drafter=None,
        pos0: int = 0,
        output_head: bool = True,
    ):
        _cache_mod.ensure_vision_encoded(m, pos0, log_fn)
        ids = list(prompt_ids)
        n = len(ids)
        if n == 0:
            raise ValueError("prompt must contain at least one token")
        step = _gen._prefill_step(n, prefill_chunk)
        state_only = getattr(m, "forward_prefill_chunk", m.forward_argmax)
        start = 0
        t_total = time.perf_counter()
        vctx = getattr(m, "_vision_ctx", None)
        if n > step:
            log_fn(f"[prefill] start {n} new tok (pos {pos0}..{pos0 + n}) | chunk={step}")
        while True:
            t_chunk = time.perf_counter()
            act_mem = mx.get_active_memory()
            if act_mem > int(15.2 * (1 << 30)):
                cur_step = min(step, 256)
            elif act_mem > int(14.2 * (1 << 30)):
                cur_step = min(step, 512)
            elif step >= 512 and (n - start) >= 1024:
                cur_step = max(step, 1024)
            else:
                cur_step = step
            end = min(start + cur_step, n)
            last = end >= n
            if last and output_head:
                fwd = m.forward_ordered_logits if ordered else m.forward_argmax
            else:
                fwd = state_only
            if vctx is not None:
                vctx["chunk_pos0"] = pos0 + start
            try:
                if tap_visits:
                    out, cache, taps = fwd(ids[start:end], cache=cache, tap_visits=tap_visits)
                else:
                    out, cache = fwd(ids[start:end], cache=cache)
                    taps = None
            finally:
                if vctx is not None:
                    vctx["chunk_pos0"] = None
            if last:
                if not output_head:
                    last_pending = list(cache.state)
                    if taps is not None:
                        last_pending.append(taps)
                    mx.eval(*last_pending)
                    if mx.get_active_memory() > (12 << 30):
                        mx.clear_cache()
                if n > step:
                    elapsed = max(time.perf_counter() - t_total, 1e-9)
                    log_fn(
                        f"[prefill] {n}/{n} tok (100.0%) | avg {n / elapsed:.0f} tok/s | "
                        f"active {mx.get_active_memory() / 1e9:.2f} GB | peak {mx.get_peak_memory() / 1e9:.2f} GB"
                    )
                if output_head and vctx is not None:
                    vctx["encoded_spans"] = None
                return out, cache, taps, pos0 + start
            pending = list(cache.state)
            if drafter is not None and taps is not None:
                drafter.push_context(taps, start_pos=pos0 + start)
                pending.extend(
                    x for x in (*drafter.ctx_k, *drafter.ctx_v, drafter.f_rnn) if x is not None
                )
            mx.eval(*pending)
            if (mx.get_active_memory() + mx.get_cache_memory()) > (16 << 30):
                mx.clear_cache()
            dt = max(time.perf_counter() - t_chunk, 1e-9)
            chunk_n = end - start
            log_fn(
                f"[prefill] {end}/{n} tok ({100.0 * end / n:.1f}%) | "
                f"{chunk_n / dt:.0f} tok/s | active {mx.get_active_memory() / 1e9:.2f} GB | "
                f"peak {mx.get_peak_memory() / 1e9:.2f} GB"
            )
            start = end

    _gen._prefill = _instrumented_prefill
    _gen.PromptCache = _cache_mod.SmartPromptCache
    _inkling.PromptCache = _cache_mod.SmartPromptCache

    orig_load = _server.Engine._load

    def _instrumented_load(self, args) -> None:
        orig_load(self, args)
        if self.prompt_cache is not None and "user" in self.ids:
            self.prompt_cache.user_tok_id = self.ids["user"]
        img_tok = self.hf.convert_tokens_to_ids("<|unused_200054|>")
        ci_tok = self.hf.convert_tokens_to_ids("<|content_image|>")
        if isinstance(img_tok, int) and img_tok >= 0:
            self.ids["image"] = img_tok
        if isinstance(ci_tok, int) and ci_tok >= 0:
            self.ids["content_image"] = ci_tok
        self.m.embed = _vision.VisionAwareEmbed(self.m.embed, self.m)
        self.m._vision_ctx = None
        if cfg.vision_path is not None and cfg.vision_path.is_file():
            sz_mb = cfg.vision_path.stat().st_size / 1e6
            log_fn(f"vision enabled: {cfg.vision_path} ({sz_mb:.1f} MB BF16, lazy on-demand mmap)")
        else:
            log_fn("vision disabled (pass --vision /path/to/vision_tower.safetensors to enable)")
        if cfg.keep_wired:
            max_rec = int(mx.device_info().get("max_recommended_working_set_size", 0) or 0)
            if max_rec > 0:
                active = int(mx.get_active_memory())
                safe_wire_cap = max_rec - (6 << 30) if max_rec <= (24 << 30) else max_rec - (4 << 30)
                target_wire = max(0, min(active + (2 << 30), safe_wire_cap))
                mx.set_wired_limit(target_wire)
                atexit.register(lambda: mx.set_wired_limit(0))
                log_fn(
                    f"wired memory pinned: {target_wire / 1e9:.2f} GB "
                    f"(active {active / 1e9:.2f} GB, max_rec {max_rec / 1e9:.2f} GB)"
                )

    _server.Engine._load = _instrumented_load

    def _apply_template_with_vision(self, req: dict, *, tokenize: bool):
        vision_ok = cfg.vision_path is not None and cfg.vision_path.is_file() and "image" in self.ids
        messages, images_raw = normalize_messages_with_vision(
            req.get("messages"),
            vision_enabled=vision_ok,
            request_error_cls=_server.RequestError,
        )
        tools, tool_names = _server.request_tools(req)
        effort = req.get("reasoning_effort")
        if isinstance(effort, dict):
            effort = effort.get("effort")
        if isinstance(effort, str):
            effort = effort.strip().lower()
            if effort not in _server.EFFORTS:
                raise _server.RequestError(400, "unknown reasoning_effort")
        else:
            effort = _server._number(
                {"reasoning_effort": effort}, "reasoning_effort", self.args.reasoning_effort, low=0, high=0.99
            )
        if (
            cfg.max_reasoning_effort in _server.EFFORTS
            and isinstance(effort, str)
            and effort in _server.EFFORTS
            and _server.EFFORTS.index(effort) > _server.EFFORTS.index(cfg.max_reasoning_effort)
        ):
            effort = cfg.max_reasoning_effort

        loop_guard_info = None
        if tokenize and cfg.loop_guard:
            loop_guard_info = _parser_mod.detect_agent_tool_loop(messages)
            if loop_guard_info is not None:
                messages = _parser_mod.inject_tool_loop_reminder(messages, loop_guard_info)
                log_fn(
                    f"[loop-guard] agent tool loop detected ({loop_guard_info['tool_name']}, "
                    f"exact={loop_guard_info['exact_streak']}, stagnant={loop_guard_info['stagnant_streak']}); "
                    f"injected reminder"
                    + (" + hard circuit-breaker armed" if (loop_guard_info["hard_block_sig"] or loop_guard_info["hard_block_tool"]) else "")
                )

        add_prompt = req.get("add_generation_prompt", True)
        if type(add_prompt) is not bool:
            raise _server.RequestError(400, "add_generation_prompt must be a boolean")
        kw: dict[str, Any] = dict(tokenize=tokenize, add_generation_prompt=add_prompt, reasoning_effort=effort)
        if tools:
            kw["tools"] = tools
        try:
            rendered = self.tok.apply_chat_template(messages, **kw)
        except Exception as e:
            raise _server.RequestError(400, "chat template could not render messages and tools") from e

        pin_hint: int | None = None
        if tokenize:
            enc_ids = rendered["input_ids"] if (isinstance(rendered, dict) or hasattr(rendered, "input_ids")) else rendered
            preamble_k = _cache_mod.count_preamble_messages(messages)
            if 0 < preamble_k < len(messages) and len(enc_ids) >= cfg.pin_min_tokens:
                try:
                    pkw: dict[str, Any] = dict(tokenize=True, add_generation_prompt=False, reasoning_effort=effort)
                    if tools:
                        pkw["tools"] = tools
                    pref = self.tok.apply_chat_template(messages[:preamble_k], **pkw)
                    if isinstance(pref, dict) or hasattr(pref, "input_ids"):
                        pref = pref["input_ids"]
                    pref_list = [int(x) for x in pref]
                    if len(pref_list) >= cfg.pin_min_tokens and [int(x) for x in enc_ids[: len(pref_list)]] == pref_list:
                        pin_hint = len(pref_list)
                except Exception:
                    pin_hint = None

        return rendered, effort, tool_names, images_raw, pin_hint, loop_guard_info

    def _render_with_vision(self, req: dict) -> str:
        return self._apply_template_with_vision(req, tokenize=False)[0]

    def _prepare_with_vision(self, req: dict) -> dict:
        enc, effort, tool_names, images_raw, pin_hint, loop_guard_info = self._apply_template_with_vision(
            dict(req, add_generation_prompt=True), tokenize=True
        )
        if isinstance(enc, dict) or hasattr(enc, "input_ids"):
            enc = enc["input_ids"]
        ids = [int(x) for x in enc]

        images_plan: list[tuple[int, int, _vision.PreprocessedImage]] = []
        cache_ids: list[int] = ids

        if images_raw:
            img_tok_id = self.ids["image"]
            ci_tok_id = self.ids.get("content_image")
            end_tok_id = self.ids["end"]

            placeholder_positions = [
                i
                for i, t in enumerate(ids)
                if t == img_tok_id
                and (ci_tok_id is None or (i > 0 and ids[i - 1] == ci_tok_id))
                and (i + 1 < len(ids) and ids[i + 1] == end_tok_id)
            ]
            if len(placeholder_positions) != len(images_raw):
                placeholder_positions = [i for i, t in enumerate(ids) if t == img_tok_id]
            if len(placeholder_positions) != len(images_raw):
                raise _server.RequestError(
                    400,
                    f"image placeholder count mismatch (template={len(placeholder_positions)}, images={len(images_raw)})",
                )

            prep_images: list[_vision.PreprocessedImage] = []
            allow_file = _server._is_loopback(self.args.host)
            for idx, part in enumerate(images_raw):
                try:
                    raw_bytes, digest = _vision.load_image_bytes(
                        part,
                        allow_file=allow_file,
                        allow_remote=cfg.allow_remote_images,
                    )
                    prep = _vision.preprocess_image(
                        raw_bytes,
                        img_hash=digest,
                        max_pixels=cfg.vision_max_pixels,
                        max_side=cfg.vision_max_side,
                    )
                except Exception as e:
                    raise _server.RequestError(400, f"failed to decode image #{idx + 1}: {e}") from e
                prep_images.append(prep)

            expanded_ids: list[int] = []
            expanded_cache_ids: list[int] = []
            prev = 0
            for pos, prep in zip(placeholder_positions, prep_images):
                expanded_ids.extend(ids[prev:pos])
                expanded_cache_ids.extend(ids[prev:pos])
                start_pos = len(expanded_ids)
                n_tok = prep.num_tokens
                expanded_ids.extend([img_tok_id] * n_tok)
                expanded_cache_ids.extend(_cache_mod.make_image_cache_ids(prep.img_hash, n_tok))
                images_plan.append((start_pos, start_pos + n_tok, prep))
                prev = pos + 1
            expanded_ids.extend(ids[prev:])
            expanded_cache_ids.extend(ids[prev:])
            ids = expanded_ids
            cache_ids = expanded_cache_ids

        ctx = self.args.context
        if not ids:
            raise _server.RequestError(400, "chat template produced no tokens")
        if len(ids) >= ctx:
            raise _server.RequestError(
                400,
                f"the request exceeds the available context size, try increasing it (prompt {len(ids)} tokens, context {ctx})",
                "exceed_context_size_error",
            )
        max_tokens = _server._number(req, "max_completion_tokens", None, low=1, integer=True)
        if max_tokens is None:
            max_tokens = _server._number(req, "max_tokens", self.args.max_tokens, low=1, integer=True)
        return {
            "ids": ids,
            "cache_ids": cache_ids,
            "images_plan": images_plan,
            "pin_hint": pin_hint,
            "loop_guard": loop_guard_info,
            "effort": effort,
            "tool_names": tool_names,
            "max_tokens": min(max_tokens, ctx - len(ids)),
            "temp": _server._number(req, "temperature", self.args.temp, low=0, high=2),
            "top_p": _server._number(req, "top_p", self.args.top_p, low=0, high=1),
            "top_k": min(_server._number(req, "top_k", 0, low=0, integer=True), self.m.lm_head.order_mx.size),
            "use_cache": req.get("cache_prompt", True) is not False,
        }

    _server.Engine._apply_template_with_vision = _apply_template_with_vision
    _server.Engine.render = _render_with_vision
    _server.Engine.prepare = _prepare_with_vision

    orig_run = _server.Engine._run

    def _instrumented_run(self, job, emit, gone):
        t0 = time.perf_counter()
        n_prompt = len(job["ids"])
        n_imgs = len(job.get("images_plan") or ())
        img_note = f" | images={n_imgs}" if n_imgs else ""
        log_fn(
            f"[req] start: prompt={n_prompt} tok{img_note} | max_tokens={job['max_tokens']} | "
            f"effort={job['effort']} | temp={job['temp']} | tools={len(job['tool_names'])}"
        )
        first_t = [None]

        def wrapped_emit(kind, payload):
            if first_t[0] is None:
                now = time.perf_counter()
                first_t[0] = now
                log_fn(
                    f"[gen] first output ({kind}) after {now - t0:.1f}s | "
                    f"active {mx.get_active_memory() / 1e9:.2f} GB"
                )
            return emit(kind, payload)

        self.m._vision_ctx = {
            "cache_ids": job.get("cache_ids", job["ids"]),
            "images_plan": job.get("images_plan") or [],
            "pin_hint": job.get("pin_hint"),
            "weights_path": cfg.vision_path,
            "encoded_spans": None,
            "chunk_pos0": None,
        }
        _parser_mod.set_active_loop_guard(job.get("loop_guard"))
        try:
            return orig_run(self, job, wrapped_emit, gone)
        finally:
            _parser_mod.set_active_loop_guard(None)
            self.m._vision_ctx = None

    _server.Engine._run = _instrumented_run
    _server.make_handler = _queue_mod.make_queued_handler_factory(_server.make_handler, log_fn)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    cfg, lamq_argv = parse_extra_args(raw_argv)
    install_patches(cfg)
    from runtime_mlx import server as _server

    return _server.main(lamq_argv)


if __name__ == "__main__":
    raise SystemExit(main())
