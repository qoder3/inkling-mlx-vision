"""Vision-safe SmartPromptCache with dynamic preamble pinning for multi-chat agent IDEs."""

from __future__ import annotations

import time
from typing import Any, Callable

import mlx.core as mx
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


class SmartPromptCache(_gen.PromptCache):
    """Multi-chat PromptCache with image-hash safety, high text min_tokens, and 1 dynamic pinned preamble slot."""

    user_tok_id: int | None = None
    default_min_tokens: int = 2048
    pin_min_tokens: int = 4096
    image_min_tokens: int = 64
    log_fn: Callable[[str], None] = staticmethod(lambda _msg: None)

    def __init__(self, max_entries: int = 3, min_tokens: int | None = None):
        super().__init__(
            max_entries=max_entries,
            min_tokens=self.default_min_tokens if min_tokens is None else min_tokens,
        )

    def _has_pinned(self, tokens: list[int], drafting: bool) -> bool:
        for e in self._entries:
            if e.get("pinned") and (e["drafter"] is not None) == drafting and e["tokens"] == tokens:
                return True
        return False

    def _store(
        self,
        tokens: list[int],
        cache: Any,
        drafter: Any = None,
        *,
        pinned: bool = False,
        min_tok: int | None = None,
    ) -> None:
        thresh = (self.pin_min_tokens if pinned else self.min_tokens) if min_tok is None else min_tok
        if self.max_entries <= 0 or len(tokens) < thresh:
            return
        n = len(tokens)
        self._clock += 1
        entry = {
            "tokens": list(tokens),
            "cache": cache.clone(),
            "drafter": drafter.context_state() if drafter is not None else None,
            "used": self._clock,
            "pinned": pinned,
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

        hit = self._match(key_ids, drafting)
        start = len(hit["tokens"]) if hit is not None else 0

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

        if hit is not None:
            self._clock += 1
            hit["used"] = self._clock
            kind = "pinned-sys" if hit.get("pinned") else "chat"
            move_hit = not hit.get("pinned") and start < n - 1 and pin_pos is None and (n - 1) >= eff_min
            if move_hit:
                # Transfer ownership in-place (this chat snapshot is about to be superseded at n - 1),
                # cutting KV copy traffic and peak RAM in half while keeping _entries clean on abort.
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

        ensure_vision_encoded(m, start, self.log_fn)

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
                start = pin_pos

            if start < n - 1:
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
                self._store(key_ids[: n - 1], cache, drafter, pinned=False, min_tok=eff_min)
                start = n - 1

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
        finally:
            cache._reserve_cap = 0
            if drafting:
                drafter._reserve_cap = 0
        return out, cache, taps, taps_start, (len(hit["tokens"]) if hit is not None else 0)
