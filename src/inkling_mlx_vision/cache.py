"""Vision-safe SmartPromptCache with dynamic preamble pinning for multi-chat agent IDEs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from runtime_mlx.inkling import generate as _gen

from . import vision as _vision


def count_preamble_messages(messages: list[dict[str, Any]]) -> int:
    """Count leading shared preamble messages (`system` + leading text-only `<system-reminder>` blocks)."""
    k = 0
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            k += 1
        elif role == "user" and isinstance(content, str) and content.lstrip().startswith("<system-reminder>"):
            k += 1
        else:
            break
    return k


def make_image_cache_ids(img_hash: str, num_tokens: int) -> list[int]:
    """Deterministic negative surrogate token IDs uniquely identifying each `(image_hash, patch_index)`."""
    h_int = int(img_hash[:15], 16)
    base = h_int << 20
    return [-((base | j) + 1) for j in range(num_tokens)]


def ensure_vision_encoded(m: Any, start_pos: int, log_fn: Callable[[str], None]) -> None:
    """Encode only images not already covered by the matched KV cache prefix (`end > start_pos`)."""
    vctx = getattr(m, "_vision_ctx", None)
    if vctx is None or vctx.get("encoded_spans") is not None:
        return
    plan = vctx.get("images_plan")
    if not plan:
        vctx["encoded_spans"] = []
        return

    needed = [(s, e, prep) for (s, e, prep) in plan if e > start_pos]
    if not needed:
        vctx["encoded_spans"] = []
        log_fn(f"[vision] all {len(plan)} image(s) served from KV cache (0ms ViT)")
        return

    weights_path = vctx.get("weights_path")
    if weights_path is None:
        raise RuntimeError("images present in request, but vision_tower weights are not configured")

    t0 = time.perf_counter()
    embs = _vision.encode_images([prep for _, _, prep in needed], weights_path)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    vctx["encoded_spans"] = [(s, e, emb) for (s, e, _), emb in zip(needed, embs)]
    tot_tok = sum(prep.num_tokens for _, _, prep in needed)
    desc = ", ".join(f"{prep.proc_size[0]}x{prep.proc_size[1]}({prep.num_tokens}t)" for _, _, prep in needed)
    log_fn(
        f"[vision] encoded {len(needed)}/{len(plan)} image(s) [{desc}] ({tot_tok} tok) in {dt_ms:.0f}ms | "
        f"active {mx.get_active_memory() / 1e9:.2f} GB"
    )


def _trim_sliding_slots_in_place(cache: Any) -> Any:
    """Trim any sliding-window KV slot with `kv_len > W` down to its last `W` active tokens (`83.9 MB` across 40 slots)."""
    if not getattr(cache, "_fast", False) or not hasattr(cache, "kv_is_sliding") or getattr(cache, "_spec_recording", False):
        return cache
    W = int(getattr(getattr(cache, "cfg", None), "sliding_window_size", 1024) or 1024)
    for s, is_sl in enumerate(cache.kv_is_sliding):
        if is_sl and cache.kv_f16[s] is not None and cache.kv_len[s] > W:
            cur = int(cache.kv_len[s])
            shift = cur - W
            cache.kv_f16[s] = mx.array(cache.kv_f16[s][:, :, shift:cur, :])
            cache.kv_len[s] = W
            cache.kv_cap[s] = W
            cache.kv_start[s] = int(cache.kv_start[s]) + shift
    return cache


def _clone_cache_for_store(cache: Any, *, share_global: bool, defer_global: bool = False) -> Any:
    """Clone `cache` for PromptCache storage.

    When `share_global=True` (storing a non-pinned chat snapshot at `n - 1` right before the final token and
    generation), share the append-only global KV buffers (`kv_u8`, `kv_q4_sc`, non-sliding `kv_f16`) zero-copy
    and only deep-copy the small fixed-size sliding-window KV buffers (`W=1024`) and 3-tap `conv_states`.
    When `defer_global=True` (while `cache` is still actively being appended to during the final prompt token
    or mid-turn `pre_tool_gen`), store `None` for non-sliding global KV slots until `finalize_post_gen` attaches
    them — keeping `cache.kv_u8` `refcount == 1` so MLX slice updates never trigger Copy-on-Write clones!
    """
    if (
        not share_global
        or not getattr(cache, "_fast", False)
        or not hasattr(cache, "kv_is_sliding")
        or getattr(cache, "_spec_recording", False)
    ):
        return _trim_sliding_slots_in_place(cache.clone())
    new = object.__new__(type(cache))
    new.__dict__.update(cache.__dict__)
    n_slots = len(cache.keys)
    cp = lambda a: None if a is None else mx.array(a)  # noqa: E731
    gl = (lambda a: None) if defer_global else (lambda a: a)  # noqa: E731
    new.keys = [cp(a) for a in cache.keys]
    new.values = [cp(a) for a in cache.values]
    new.kv_f16 = [cp(a) if cache.kv_is_sliding[s] else gl(a) for s, a in enumerate(cache.kv_f16)]
    new.kv_u8 = [cp(a) if cache.kv_is_sliding[s] else gl(a) for s, a in enumerate(cache.kv_u8)]
    new.kv_q4_sc = [cp(a) if cache.kv_is_sliding[s] else gl(a) for s, a in enumerate(cache.kv_q4_sc)]
    new.kv_len = list(cache.kv_len)
    new.kv_start = list(cache.kv_start)
    new.kv_cap = [int(cache.kv_cap[s]) if (cache.kv_is_sliding[s] or not defer_global) else 0 for s in range(n_slots)]
    new.kv_is_sliding = list(cache.kv_is_sliding)
    new.conv_states = [[cp(a) for a in row] for row in cache.conv_states]
    new._spec_base_kv_len = list(cache._spec_base_kv_len)
    new._spec_kv_full = [None] * n_slots
    new._spec_conv_full = [[None, None, None, None] for _ in range(n_slots)]
    new._spec_conv_pair = [[None, None, None, None] for _ in range(n_slots)]
    new._reserve_cap = 0
    _trim_sliding_slots_in_place(new)
    mx.eval(*new.state)
    return new


def _clone_drafter_for_store(drafter: Any, *, share_global: bool, defer_global: bool = False) -> Any:
    if drafter is None:
        return None
    if not share_global or not hasattr(drafter, "ctx_k") or not hasattr(drafter, "local_layers"):
        return drafter.context_state()
    cp = lambda a: None if a is None else mx.array(a)  # noqa: E731
    gl = (lambda a: None) if defer_global else (lambda a: a)  # noqa: E731
    local = drafter.local_layers
    st = {
        "ctx_k": [cp(a) if i in local else gl(a) for i, a in enumerate(drafter.ctx_k)],
        "ctx_v": [cp(a) if i in local else gl(a) for i, a in enumerate(drafter.ctx_v)],
        "_ctx_cur": list(drafter._ctx_cur),
        "_ctx_cap": [
            int(drafter._ctx_cap[i]) if (i in local or not defer_global) else 0
            for i in range(len(drafter._ctx_cap))
        ],
        "ctx_len": drafter.ctx_len,
        "f_rnn": cp(drafter.f_rnn),
    }
    mx.eval(*[a for a in (*st["ctx_k"], *st["ctx_v"], st["f_rnn"]) if a is not None])
    return st


def _save_inplace_rollback(cache: Any, drafter: Any) -> dict[str, Any]:
    """Lightweight rollback state for `move_hit=True` without copying global KV buffers."""
    cp = lambda a: None if a is None else mx.array(a)  # noqa: E731
    snap: dict[str, Any] = {"seq_len": getattr(cache, "seq_len", 0)}
    if hasattr(cache, "kv_is_sliding"):
        snap["keys"] = [cp(a) for a in cache.keys]
        snap["values"] = [cp(a) for a in cache.values]
        snap["kv_f16_sliding"] = [
            cp(a) if cache.kv_is_sliding[s] else None for s, a in enumerate(cache.kv_f16)
        ]
        snap["kv_len"] = list(cache.kv_len)
        snap["kv_start"] = list(cache.kv_start)
        snap["conv_states"] = [[cp(a) for a in row] for row in cache.conv_states]
    if drafter is not None and hasattr(drafter, "ctx_k"):
        local = getattr(drafter, "local_layers", set())
        snap["drafter"] = {
            "ctx_k_local": {i: cp(drafter.ctx_k[i]) for i in local if i < len(drafter.ctx_k)},
            "ctx_v_local": {i: cp(drafter.ctx_v[i]) for i in local if i < len(drafter.ctx_v)},
            "_ctx_cur": list(drafter._ctx_cur),
            "ctx_len": drafter.ctx_len,
            "f_rnn": cp(drafter.f_rnn),
        }
    return snap


def _apply_inplace_rollback(hit: dict[str, Any], cache: Any, drafter: Any, snap: dict[str, Any]) -> None:
    """Restore `cache` and `drafter` to their pre-prefill state if an in-place (`move_hit`) prefill aborts or errors."""
    cache.seq_len = snap["seq_len"]
    if "kv_len" in snap:
        cache.keys = snap["keys"]
        cache.values = snap["values"]
        for s, a in enumerate(snap["kv_f16_sliding"]):
            if cache.kv_is_sliding[s]:
                cache.kv_f16[s] = a
                cache.kv_cap[s] = 0 if a is None else int(a.shape[2])
        cache.kv_len = list(snap["kv_len"])
        cache.kv_start = list(snap["kv_start"])
        cache.conv_states = snap["conv_states"]
    hit["cache"] = cache
    if drafter is not None and "drafter" in snap and hasattr(drafter, "ctx_k"):
        ds = snap["drafter"]
        for i, a in ds["ctx_k_local"].items():
            drafter.ctx_k[i] = a
            drafter._ctx_cap[i] = 0 if a is None else int(a.shape[2])
        for i, a in ds["ctx_v_local"].items():
            drafter.ctx_v[i] = a
        drafter._ctx_cur = list(ds["_ctx_cur"])
        drafter.ctx_len = ds["ctx_len"]
        drafter.f_rnn = ds["f_rnn"]
        hit["drafter"] = {
            "ctx_k": list(drafter.ctx_k),
            "ctx_v": list(drafter.ctx_v),
            "_ctx_cur": list(drafter._ctx_cur),
            "_ctx_cap": list(drafter._ctx_cap),
            "ctx_len": drafter.ctx_len,
            "f_rnn": drafter.f_rnn,
        }


def _sync_shared_global_buffers(dst_cache: Any, dst_drafter: Any, src_cache: Any, src_drafter: Any) -> None:
    """Point `dst_cache`/`dst_drafter` non-sliding global KV slots at `src_cache`/`src_drafter`'s arrays.

    When `src_cache` grows across a 4096-token boundary during decode, its new global KV array contains
    the exact same `0 .. dst_cache.seq_len` prefix, so syncing the reference frees the smaller old buffer.
    """
    if (
        dst_cache is not None
        and src_cache is not None
        and getattr(dst_cache, "_fast", False)
        and hasattr(dst_cache, "kv_is_sliding")
    ):
        for s, is_sl in enumerate(dst_cache.kv_is_sliding):
            if not is_sl:
                if src_cache.kv_f16[s] is not None:
                    dst_cache.kv_f16[s] = src_cache.kv_f16[s]
                    dst_cache.kv_cap[s] = int(src_cache.kv_f16[s].shape[2])
                if src_cache.kv_u8[s] is not None:
                    dst_cache.kv_u8[s] = src_cache.kv_u8[s]
                    dst_cache.kv_cap[s] = int(src_cache.kv_u8[s].shape[2])
                if src_cache.kv_q4_sc[s] is not None:
                    dst_cache.kv_q4_sc[s] = src_cache.kv_q4_sc[s]
    if (
        isinstance(dst_drafter, dict)
        and src_drafter is not None
        and hasattr(src_drafter, "ctx_k")
        and hasattr(src_drafter, "local_layers")
    ):
        local = src_drafter.local_layers
        for i in range(len(src_drafter.ctx_k)):
            if i not in local and src_drafter.ctx_k[i] is not None:
                dst_drafter["ctx_k"][i] = src_drafter.ctx_k[i]
                dst_drafter["ctx_v"][i] = src_drafter.ctx_v[i]
                dst_drafter["_ctx_cap"][i] = int(src_drafter.ctx_k[i].shape[2])


def _pinned_cache_identity(m: Any, tokens: list[int], drafting: bool) -> dict[str, Any]:
    packed_dir = str(getattr(m, "packed_dir", ""))
    cfg = getattr(m, "cfg", None)
    kv_mode = str(getattr(cfg, "global_kv_mode", "fp8"))
    tok_sha = hashlib.sha256(np.asarray(tokens, dtype=np.int32).tobytes()).hexdigest()
    return {
        "packed_dir": packed_dir,
        "global_kv_mode": kv_mode,
        "drafting": bool(drafting),
        "num_tokens": len(tokens),
        "tokens_sha256": tok_sha,
    }


class SmartPromptCache(_gen.PromptCache):
    """Multi-chat PromptCache with image-hash safety, high text min_tokens, and 1 dynamic pinned preamble slot."""

    user_tok_id: int | None = None
    default_min_tokens: int = 2048
    pin_min_tokens: int = 4096
    image_min_tokens: int = 64
    disk_cache_dir: Path | None = (
        Path(os.environ["INKLING_PIN_CACHE_DIR"]).expanduser()
        if os.environ.get("INKLING_PIN_CACHE_DIR")
        else (
            None
            if os.environ.get("INKLING_PIN_CACHE_DIR") == ""
            else Path.home() / ".cache" / "inkling-mlx-vision"
        )
    )
    log_fn: Callable[[str], None] = staticmethod(lambda _msg: None)

    def __init__(self, max_entries: int = 3, min_tokens: int | None = None):
        super().__init__(
            max_entries=max_entries,
            min_tokens=self.default_min_tokens if min_tokens is None else min_tokens,
        )
        self._active_gen: dict[str, Any] | None = None
        self._cached_ngram_idx: Any = None

    def clear(self) -> None:
        super().clear()
        self._active_gen = None
        self._cached_ngram_idx = None

    def _has_pinned(self, tokens: list[int], drafting: bool) -> bool:
        for e in self._entries:
            if e.get("pinned") and (e["drafter"] is not None) == drafting and e["tokens"] == tokens:
                return True
        return False

    def _flush_active_gen_global_buffers(self) -> None:
        """Attach deferred global KV buffers from `_active_gen` to `entry` and `pre_tool_gen` once writing finishes."""
        ag = self._active_gen
        if not isinstance(ag, dict):
            return
        entry = ag.get("entry")
        cache = ag.get("cache")
        drafter = ag.get("drafter")
        if isinstance(entry, dict) and cache is not None:
            _sync_shared_global_buffers(entry.get("cache"), entry.get("drafter"), cache, drafter)
            pt = entry.get("pre_tool_gen")
            if isinstance(pt, dict):
                _sync_shared_global_buffers(pt.get("cache"), pt.get("drafter"), cache, drafter)

    def _match(self, ids: list[int], drafting: bool) -> dict[str, Any] | None:
        """Match the longest cached prefix across `post_gen`, `pre_tool_gen`, and base prompt snapshots."""
        self._flush_active_gen_global_buffers()
        best: dict[str, Any] | None = None
        best_tier = "base"
        best_len = 0
        n_ids = len(ids)
        for e in self._entries:
            if (e["drafter"] is not None) != drafting:
                continue
            cand_tier = None
            cand_len = 0
            pg = e.get("post_gen")
            pt = e.get("pre_tool_gen")
            if pg is not None and len(pg["tokens"]) < n_ids and ids[: len(pg["tokens"])] == pg["tokens"]:
                cand_tier = "post_gen"
                cand_len = len(pg["tokens"])
            elif pt is not None and len(pt["tokens"]) < n_ids and ids[: len(pt["tokens"])] == pt["tokens"]:
                cand_tier = "pre_tool_gen"
                cand_len = len(pt["tokens"])
            elif len(e["tokens"]) < n_ids and ids[: len(e["tokens"])] == e["tokens"]:
                cand_tier = "base"
                cand_len = len(e["tokens"])
            if cand_tier is not None and (best is None or cand_len > best_len):
                best = e
                best_tier = cand_tier
                best_len = cand_len

        if best is not None and not best.get("pinned"):
            if best_tier == "post_gen":
                pg = best["post_gen"]
                best["tokens"] = pg["tokens"]
                best["cache"] = pg["cache"]
                best["drafter"] = pg["drafter"]
                best["_promoted_gen_toks"] = int(pg.get("gen_toks", 0))
            elif best_tier == "pre_tool_gen":
                pt = best["pre_tool_gen"]
                best["tokens"] = pt["tokens"]
                best["cache"] = pt["cache"]
                best["drafter"] = pt["drafter"]
                best["_promoted_gen_toks"] = int(pt.get("gen_toks", 0))
            else:
                best["_promoted_gen_toks"] = 0
            best["post_gen"] = None
            best["pre_tool_gen"] = None
        return best

    def acquire_ngram_index(self, initial_tokens) -> Any:
        """Reuse the previous turn's `NgramDraftIndex` in O(delta) when its tokens are a prefix of `initial_tokens`."""
        ng = self._cached_ngram_idx
        self._cached_ngram_idx = None
        if ng is None or not hasattr(ng, "tokens"):
            return None
        ids = [int(t) for t in initial_tokens]
        old_toks = ng.tokens
        n_old = len(old_toks)
        n_new = len(ids)
        if 0 < n_old <= n_new and ids[:n_old] == old_toks:
            ng.extend(ids[n_old:])
            return ng
        if 1 < n_old and (n_old - 1) <= n_new and ids[: n_old - 1] == old_toks[:-1]:
            del old_toks[-1:]
            ng._indexed_end = min(ng._indexed_end, len(old_toks))
            ng.extend(ids[n_old - 1 :])
            return ng
        return None

    def checkpoint_pre_tool_gen(self, ngram_idx: Any) -> None:
        """Save a thinking-end checkpoint right at `<|content_invoke_tool_json|>` before any JSON argument keys."""
        ag = self._active_gen
        if not isinstance(ag, dict):
            return
        entry = ag.get("entry")
        cache = ag.get("cache")
        drafter = ag.get("drafter")
        key_ids = ag.get("key_ids")
        prompt_len = int(ag.get("prompt_len", 0))
        if (
            entry is None
            or cache is None
            or key_ids is None
            or entry.get("pre_tool_gen") is not None
            or getattr(cache, "_spec_recording", False)
        ):
            return
        seq_len = int(getattr(cache, "seq_len", -1))
        if seq_len != len(ngram_idx.tokens) - 1 or seq_len <= prompt_len:
            return
        if drafter is not None and hasattr(drafter, "ctx_len") and int(drafter.ctx_len) != seq_len:
            return
        gen_committed = [int(t) for t in ngram_idx.tokens[prompt_len:-1]]
        if len(gen_committed) < 32:
            return
        # Defer global KV buffer attachment until `finalize_post_gen` so `cache.kv_u8` keeps `refcount == 1`
        # throughout tool-JSON generation (0 MB Copy-on-Write!).
        entry["pre_tool_gen"] = {
            "tokens": list(key_ids) + gen_committed,
            "cache": _clone_cache_for_store(cache, share_global=True, defer_global=True),
            "drafter": _clone_drafter_for_store(drafter, share_global=True, defer_global=True),
            "gen_toks": len(gen_committed),
        }

    def finalize_post_gen(self, ngram_idx: Any) -> None:
        """Attach the post-generation KV cache + drafter state to the active chat entry (0 MB extra Global KV)."""
        self._cached_ngram_idx = ngram_idx
        self._flush_active_gen_global_buffers()
        ag = self._active_gen
        self._active_gen = None
        if not isinstance(ag, dict):
            return
        entry = ag.get("entry")
        cache = ag.get("cache")
        drafter = ag.get("drafter")
        key_ids = ag.get("key_ids")
        prompt_len = int(ag.get("prompt_len", 0))
        if entry is None or cache is None or key_ids is None or getattr(cache, "_spec_recording", False):
            return
        seq_len = int(getattr(cache, "seq_len", -1))
        if seq_len != len(ngram_idx.tokens) - 1 or seq_len <= prompt_len:
            return
        if drafter is not None and hasattr(drafter, "ctx_len") and int(drafter.ctx_len) != seq_len:
            return
        gen_committed = [int(t) for t in ngram_idx.tokens[prompt_len:-1]]
        if not gen_committed:
            return
        pt = entry.get("pre_tool_gen")
        if isinstance(pt, dict) and len(pt.get("tokens", ())) == len(key_ids) + len(gen_committed):
            entry["pre_tool_gen"] = None
        entry["post_gen"] = {
            "tokens": list(key_ids) + gen_committed,
            "cache": _clone_cache_for_store(cache, share_global=True),
            "drafter": _clone_drafter_for_store(drafter, share_global=True),
            "gen_toks": len(gen_committed),
        }

    def _save_pinned_to_disk(self, m: Any, tokens: list[int], cache: Any, drafter: Any) -> None:
        """Persist the pinned shared system/tools KV snapshot to disk (`~620 MB`) for instant cold-start reloads."""
        cdir = self.disk_cache_dir
        if cdir is None or not getattr(cache, "_fast", False) or not hasattr(cache, "kv_is_sliding"):
            return
        try:
            t0 = time.perf_counter()
            cdir.mkdir(parents=True, exist_ok=True)
            drafting = drafter is not None and hasattr(drafter, "ctx_k")
            meta = _pinned_cache_identity(m, tokens, drafting)
            tensors: dict[str, mx.array] = {}
            for s in range(len(cache.keys)):
                if cache.kv_f16[s] is not None:
                    tensors[f"kv_f16.{s}"] = cache.kv_f16[s]
                if cache.kv_u8[s] is not None:
                    tensors[f"kv_u8.{s}"] = cache.kv_u8[s]
                if cache.kv_q4_sc[s] is not None:
                    tensors[f"kv_q4_sc.{s}"] = cache.kv_q4_sc[s]
                for ci, c_arr in enumerate(cache.conv_states[s]):
                    if c_arr is not None:
                        tensors[f"conv.{s}.{ci}"] = c_arr
            meta["seq_len"] = int(cache.seq_len)
            meta["kv_len"] = [int(x) for x in cache.kv_len]
            meta["kv_start"] = [int(x) for x in cache.kv_start]
            meta["kv_cap"] = [int(x) for x in cache.kv_cap]
            meta["kv_is_sliding"] = [bool(x) for x in cache.kv_is_sliding]
            meta["tokens"] = [int(x) for x in tokens]
            if drafting:
                for i in range(len(drafter.ctx_k)):
                    if drafter.ctx_k[i] is not None:
                        tensors[f"d_k.{i}"] = drafter.ctx_k[i]
                    if drafter.ctx_v[i] is not None:
                        tensors[f"d_v.{i}"] = drafter.ctx_v[i]
                if drafter.f_rnn is not None:
                    tensors["d_rnn"] = drafter.f_rnn
                meta["d_cur"] = [int(x) for x in drafter._ctx_cur]
                meta["d_cap"] = [int(x) for x in drafter._ctx_cap]
                meta["d_len"] = int(drafter.ctx_len)
            st_path = cdir / "pinned_sys.safetensors"
            json_path = cdir / "pinned_sys.json"
            mx.save_safetensors(str(st_path), tensors)
            json_path.write_text(json.dumps(meta), encoding="utf-8")
            dt_ms = (time.perf_counter() - t0) * 1000.0
            sz_mb = st_path.stat().st_size / 1e6
            self.log_fn(f"[cache] saved pinned-sys snapshot to disk ({sz_mb:.0f} MB in {dt_ms:.0f}ms)")
        except Exception as e:
            self.log_fn(f"[cache] note: skipped saving pinned-sys to disk: {e}")

    def _try_load_pinned_from_disk(self, m: Any, key_ids: list[int], drafting: bool, drafter: Any) -> bool:
        """Load the persisted `pinned-sys` snapshot from disk if it matches the model and `key_ids` prefix."""
        cdir = self.disk_cache_dir
        if cdir is None or any(e.get("pinned") for e in self._entries):
            return False
        json_path = cdir / "pinned_sys.json"
        st_path = cdir / "pinned_sys.safetensors"
        if not json_path.is_file() or not st_path.is_file():
            return False
        try:
            t0 = time.perf_counter()
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            pin_len = int(meta.get("num_tokens", 0))
            if pin_len < self.pin_min_tokens or pin_len >= len(key_ids) - 1:
                return False
            saved_tokens = [int(x) for x in meta.get("tokens", ())]
            if len(saved_tokens) != pin_len or key_ids[:pin_len] != saved_tokens:
                return False
            want_id = _pinned_cache_identity(m, saved_tokens, drafting)
            for k in ("packed_dir", "global_kv_mode", "drafting", "num_tokens", "tokens_sha256"):
                if meta.get(k) != want_id.get(k):
                    return False
            cache = m.make_cache()
            if not getattr(cache, "_fast", False) or not hasattr(cache, "kv_is_sliding"):
                return False
            tensors = mx.load(str(st_path))
            n_slots = len(cache.keys)
            cache.seq_len = int(meta["seq_len"])
            cache.kv_len = [int(x) for x in meta["kv_len"]]
            cache.kv_start = [int(x) for x in meta["kv_start"]]
            cache.kv_cap = [int(x) for x in meta["kv_cap"]]
            cache.kv_is_sliding = [bool(x) for x in meta["kv_is_sliding"]]
            for s in range(n_slots):
                cache.kv_f16[s] = tensors.get(f"kv_f16.{s}")
                cache.kv_u8[s] = tensors.get(f"kv_u8.{s}")
                cache.kv_q4_sc[s] = tensors.get(f"kv_q4_sc.{s}")
                for ci in range(4):
                    cache.conv_states[s][ci] = tensors.get(f"conv.{s}.{ci}")
            d_state = None
            if drafting and drafter is not None and hasattr(drafter, "ctx_k"):
                n_dl = len(drafter.ctx_k)
                d_state = {
                    "ctx_k": [tensors.get(f"d_k.{i}") for i in range(n_dl)],
                    "ctx_v": [tensors.get(f"d_v.{i}") for i in range(n_dl)],
                    "_ctx_cur": [int(x) for x in meta["d_cur"]],
                    "_ctx_cap": [int(x) for x in meta["d_cap"]],
                    "ctx_len": int(meta["d_len"]),
                    "f_rnn": tensors.get("d_rnn"),
                }
                mx.eval(*[a for a in (*d_state["ctx_k"], *d_state["ctx_v"], d_state["f_rnn"]) if a is not None])
            mx.eval(*cache.state)
            self._clock += 1
            self._entries.insert(
                0,
                {
                    "tokens": saved_tokens,
                    "cache": cache,
                    "drafter": d_state,
                    "used": self._clock,
                    "pinned": True,
                },
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.log_fn(
                f"[cache] loaded pinned-sys prefix ({pin_len} tok) from disk in {dt_ms:.0f}ms | "
                f"active {mx.get_active_memory() / 1e9:.2f} GB"
            )
            return True
        except Exception as e:
            self.log_fn(f"[cache] note: skipped loading pinned-sys from disk: {e}")
            return False

    def _store(
        self,
        tokens: list[int],
        cache: Any,
        drafter: Any = None,
        *,
        pinned: bool = False,
        min_tok: int | None = None,
        defer_global: bool = False,
    ) -> dict[str, Any] | None:
        thresh = (self.pin_min_tokens if pinned else self.min_tokens) if min_tok is None else min_tok
        if self.max_entries <= 0 or len(tokens) < thresh:
            return None
        n = len(tokens)
        self._clock += 1
        share_global = not pinned
        entry = {
            "tokens": list(tokens),
            "cache": _clone_cache_for_store(cache, share_global=share_global, defer_global=defer_global),
            "drafter": _clone_drafter_for_store(drafter, share_global=share_global, defer_global=defer_global),
            "used": self._clock,
            "pinned": pinned,
            "post_gen": None,
            "pre_tool_gen": None,
        }
        if pinned:
            # Keep at most 1 pinned shared preamble slot; replace automatically when preamble changes.
            self._entries = [e for e in self._entries if not e.get("pinned")]
            self._entries.append(entry)
        else:
            # Replace shorter non-pinned snapshots extended by this chat while preserving pinned preamble.
            self._entries = [
                e
                for e in self._entries
                if e.get("pinned")
                or not (len(e["tokens"]) <= n and tokens[: len(e["tokens"])] == e["tokens"])
            ]
            self._entries.append(entry)

        pinned_entries = [e for e in self._entries if e.get("pinned")]
        chat_entries = sorted(
            (e for e in self._entries if not e.get("pinned")),
            key=lambda e: e["used"],
            reverse=True,
        )
        max_chats = max(1, self.max_entries - len(pinned_entries))
        self._entries = pinned_entries + chat_entries[:max_chats]
        return entry if entry in self._entries else None

    def _pre_evict_for_new_chat(
        self,
        key_ids_prefix: list[int],
        hit: dict[str, Any] | None,
        will_have_pinned: bool,
    ) -> None:
        """Free superseded or soon-to-be-evicted non-pinned chat caches BEFORE allocating a new chat KV cache."""
        pinned_count = 1 if will_have_pinned else sum(1 for e in self._entries if e.get("pinned"))
        max_chats = max(1, self.max_entries - pinned_count)
        surviving_old = max(0, max_chats - 1)
        pinned_entries = [e for e in self._entries if e.get("pinned")]
        n_pref = len(key_ids_prefix)
        other_chats = [
            e
            for e in self._entries
            if not e.get("pinned")
            and (
                e is hit
                or not (len(e["tokens"]) <= n_pref and key_ids_prefix[: len(e["tokens"])] == e["tokens"])
            )
        ]
        other_chats.sort(key=lambda e: (e is hit, e["used"]), reverse=True)
        min_keep = 1 if (hit is not None and not hit.get("pinned")) else 0
        keep_chats = other_chats[: max(surviving_old, min_keep)]
        old_chat_count = sum(1 for e in self._entries if not e.get("pinned"))
        if len(keep_chats) < old_chat_count:
            self._entries = pinned_entries + keep_chats
            mx.clear_cache()

    def prefill(
        self,
        m: Any,
        prompt_ids,
        *,
        ordered: bool,
        prefill_chunk=_gen.PREFILL_CHUNK,
        tap_visits=(),
        drafter=None,
    ):
        ids = list(prompt_ids)
        n = len(ids)
        drafting = drafter is not None
        vctx = getattr(m, "_vision_ctx", None)
        has_images = bool(vctx and vctx.get("images_plan"))
        if vctx is not None and vctx.get("cache_ids") is not None and len(vctx["cache_ids"]) == n:
            key_ids = vctx["cache_ids"]
        else:
            key_ids = ids
        eff_min = self.image_min_tokens if has_images else self.min_tokens

        if self.max_entries <= 0 or n - 1 < max(1, eff_min):
            self.log_fn(f"[cache] bypass (prompt {n} < min_tokens {eff_min})")
            return _gen._prefill_prompt(
                m,
                ids,
                ordered=ordered,
                prefill_chunk=prefill_chunk,
                prompt_cache=None,
                tap_visits=tap_visits,
                drafter=drafter,
            )

        owner = (m, drafter)
        if self._owner is None or any(a is not b for a, b in zip(owner, self._owner)):
            self.clear()
            self._owner = owner

        self._flush_active_gen_global_buffers()
        self._active_gen = None
        if self.max_entries >= 2 and not any(e.get("pinned") for e in self._entries):
            self._try_load_pinned_from_disk(m, key_ids, drafting, drafter)

        hit = self._match(key_ids, drafting)
        reused_tokens = len(hit["tokens"]) if hit is not None else 0
        start = reused_tokens

        # Encode images BEFORE cloning hit["cache"] so ViT weights/activations are freed first.
        ensure_vision_encoded(m, start, self.log_fn)

        pin_pos: int | None = None
        first_img_pos = vctx["images_plan"][0][0] if has_images else n
        if self.max_entries >= 2:
            u = vctx.get("pin_hint") if vctx is not None else None
            if u is None and self.user_tok_id is not None:
                try:
                    u = ids.index(self.user_tok_id)
                except ValueError:
                    u = None
            if (
                u is not None
                and self.pin_min_tokens <= u < n - 1
                and u <= first_img_pos
                and start < u
                and not self._has_pinned(key_ids[:u], drafting)
            ):
                pin_pos = u

        move_hit = (
            hit is not None
            and not hit.get("pinned")
            and start < n - 1
            and pin_pos is None
            and (n - 1) >= eff_min
        )
        should_store_chat = (n - 1) >= eff_min
        will_have_pinned = (pin_pos is not None) or any(e.get("pinned") for e in self._entries)
        if not move_hit and start < n - 1 and should_store_chat:
            max_chats = max(1, self.max_entries - (1 if will_have_pinned else 0))
            existing_chats = [e for e in self._entries if not e.get("pinned")]
            if (
                max_chats == 1
                and hit is not None
                and hit.get("pinned")
                and pin_pos is None
                and existing_chats
            ):
                best_old = max(existing_chats, key=lambda e: len(e["tokens"]))
                old_unpinned = len(best_old["tokens"]) - (
                    start if best_old["tokens"][:start] == hit["tokens"] else 0
                )
                new_unpinned = (n - 1) - start
                if old_unpinned >= 4096 and new_unpinned < min(self.min_tokens, old_unpinned // 4):
                    should_store_chat = False
                    self.log_fn(
                        f"[cache] preserving large chat cache ({len(best_old['tokens'])} tok) "
                        f"over short side query (+{new_unpinned} tok past pinned-sys)"
                    )
            if should_store_chat:
                self._pre_evict_for_new_chat(key_ids[: n - 1], hit, will_have_pinned)

        rollback_snap: dict[str, Any] | None = None
        if hit is not None:
            self._clock += 1
            hit["used"] = self._clock
            gen_promoted = int(hit.pop("_promoted_gen_toks", 0) or 0)
            if hit.get("pinned"):
                kind = "pinned-sys"
            elif gen_promoted > 0:
                kind = f"chat+gen({gen_promoted}t)"
            else:
                kind = "chat"
            if move_hit:
                # Transfer ownership in-place (this chat snapshot is about to be superseded at n - 1),
                # cutting KV copy traffic and peak RAM in half while keeping a lightweight rollback on abort.
                self._entries = [e for e in self._entries if e is not hit]
                cache = hit["cache"]
                if drafting:
                    st = hit["drafter"]
                    if st is not None and hasattr(drafter, "ctx_k") and isinstance(st, dict) and "ctx_k" in st:
                        drafter.ctx_k = list(st["ctx_k"])
                        drafter.ctx_v = list(st["ctx_v"])
                        drafter._ctx_cur = list(st["_ctx_cur"])
                        drafter._ctx_cap = list(st["_ctx_cap"])
                        drafter.ctx_len = st["ctx_len"]
                        drafter.f_rnn = st["f_rnn"]
                    else:
                        drafter.restore_context(st)
                rollback_snap = _save_inplace_rollback(cache, drafter)
                if drafting and isinstance(hit.get("drafter"), dict):
                    # Release hit["drafter"] references to global ctx_k/ctx_v so drafter.ctx_k keeps refcount == 1!
                    hit["drafter"] = None
                self.log_fn(f"[cache] HIT {start}/{n} tok ({kind}, in-place +{n - start} tok)")
            else:
                cache = hit["cache"].clone()
                if drafting:
                    drafter.restore_context(hit["drafter"])
                self.log_fn(f"[cache] HIT {start}/{n} tok ({kind}, new {n - start} tok)")
        else:
            cache = m.make_cache()
            if drafting:
                drafter.reset()
            self.log_fn(f"[cache] MISS 0/{n} tok (entries={len(self._entries)}/{self.max_entries})")

        stored_chat: dict[str, Any] | None = (
            hit if (hit is not None and not hit.get("pinned") and not move_hit) else None
        )
        try:
            if pin_pos is not None:
                # Size tightly to pin_pos so the pinned snapshot holds no unused tail capacity.
                cache._reserve_cap = pin_pos
                if drafting:
                    drafter._reserve_cap = pin_pos
                _, cache, taps, taps_start = _gen._prefill(
                    m,
                    ids[start:pin_pos],
                    cache,
                    ordered=False,
                    prefill_chunk=prefill_chunk,
                    tap_visits=tap_visits,
                    drafter=drafter,
                    pos0=start,
                    output_head=False,
                )
                if drafting:
                    drafter.push_context(taps, start_pos=taps_start)
                self._store(key_ids[:pin_pos], cache, drafter, pinned=True)
                self.log_fn(
                    f"[cache] pinned shared system/tools prefix ({pin_pos} tok) | "
                    f"active {mx.get_active_memory() / 1e9:.2f} GB"
                )
                self._save_pinned_to_disk(m, key_ids[:pin_pos], cache, drafter)
                start = pin_pos

            if start < n - 1 and (should_store_chat or move_hit):
                # Pre-reserve full prompt + decode headroom in one allocation (eliminates O(N^2) KV copies).
                cache._reserve_cap = n + 512
                if drafting:
                    drafter._reserve_cap = n + 512
                _, cache, taps, taps_start = _gen._prefill(
                    m,
                    ids[start : n - 1],
                    cache,
                    ordered=False,
                    prefill_chunk=prefill_chunk,
                    tap_visits=tap_visits,
                    drafter=drafter,
                    pos0=start,
                    output_head=False,
                )
                if drafting:
                    drafter.push_context(taps, start_pos=taps_start)
                rollback_snap = None
                stored_chat = self._store(
                    key_ids[: n - 1], cache, drafter, pinned=False, min_tok=eff_min, defer_global=True
                )
                if stored_chat is not None:
                    self._active_gen = {
                        "entry": stored_chat,
                        "key_ids": list(key_ids),
                        "prompt_len": n,
                        "cache": cache,
                        "drafter": drafter,
                    }
                start = n - 1

            if not should_store_chat and start < n - 1:
                cache._reserve_cap = n + 256
                if drafting:
                    drafter._reserve_cap = n + 256

            out, cache, taps, taps_start = _gen._prefill(
                m,
                ids[start:],
                cache,
                ordered=ordered,
                prefill_chunk=prefill_chunk,
                tap_visits=tap_visits,
                drafter=drafter,
                pos0=start,
            )
            if stored_chat is not None:
                self._active_gen = {
                    "entry": stored_chat,
                    "key_ids": list(key_ids),
                    "prompt_len": n,
                    "cache": cache,
                    "drafter": drafter,
                }
        except BaseException:
            self._flush_active_gen_global_buffers()
            self._active_gen = None
            if move_hit and hit is not None and rollback_snap is not None:
                _apply_inplace_rollback(hit, cache, drafter, rollback_snap)
                self._entries.append(hit)
                self.log_fn(f"[cache] restored in-place chat cache ({len(hit['tokens'])} tok) after interrupted prefill")
            raise
        finally:
            cache._reserve_cap = 0
            if drafting:
                drafter._reserve_cap = 0
        return out, cache, taps, taps_start, reused_tokens
