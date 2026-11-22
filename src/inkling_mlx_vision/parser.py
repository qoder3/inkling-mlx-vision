"""Monitored OutputParser with live token/state logging and runaway repetition-loop guards."""

from __future__ import annotations

import collections
import time
from typing import Callable

import mlx.core as mx
from runtime_mlx import server as _server

_OrigOutputParser = _server.OutputParser


class MonitoredOutputParser(_OrigOutputParser):
    """Extends `lamq serve`'s `OutputParser` with visibility during tool buffering and stuck-loop protection."""

    loop_guard_enabled: bool = True
    max_discard_tool_tokens: int = 1024
    loop_window: int = 512
    loop_min_span: int = 256
    loop_min_reps: int = 6
    loop_max_period: int = 64
    log_interval_s: float = 2.0
    log_fn: Callable[[str], None] = staticmethod(lambda _msg: None)

    def __init__(self, tokenizer, ids: dict[str, int], tool_names=()):
        super().__init__(tokenizer, ids, tool_names)
        now = time.perf_counter()
        self._gen_t0 = now
        self._last_log = now
        self._n_tok = 0
        self._discard_toks = 0
        self._recent: collections.deque[int] = collections.deque(maxlen=self.loop_window)

    def detect_loop(self) -> tuple[int, int] | None:
        """Return `(period, reps)` if trailing tokens repeat verbatim for `>= loop_min_span` tokens."""
        buf = list(self._recent)
        n = len(buf)
        if n < self.loop_min_span:
            return None
        for p in range(1, self.loop_max_period + 1):
            reps = max(self.loop_min_reps, (self.loop_min_span + p - 1) // p)
            span = p * reps
            if span > n:
                continue
            unit = buf[n - p :]
            ok = True
            for k in range(2, reps + 1):
                if buf[n - k * p : n - (k - 1) * p] != unit:
                    ok = False
                    break
            if ok:
                return p, reps
        return None

    def feed(self, t: int) -> list:
        out = super().feed(t)
        self._n_tok += 1
        self._recent.append(t)

        if self.loop_guard_enabled:
            if self.state == "discard_tool":
                self._discard_toks += 1
                if self._discard_toks >= self.max_discard_tool_tokens:
                    self.log_fn(
                        f"[gen] guard: aborted stuck discard_tool ({self._discard_toks} tok, "
                        f"name={self.tool_name!r}, total={self._n_tok} tok)"
                    )
                    self.done = True
                    return out
            else:
                self._discard_toks = 0

            if not self.done and (self._n_tok & 31) == 0 and len(self._recent) >= self.loop_min_span:
                loop = self.detect_loop()
                if loop is not None:
                    p, reps = loop
                    if self.state in ("thinking", "text"):
                        self._flush(out)
                    self.log_fn(
                        f"[gen] guard: repetition loop detected (state={self.state}, period={p}, "
                        f"reps={reps}, total={self._n_tok} tok); stopping cleanly"
                    )
                    self.done = True
                    return out

        now = time.perf_counter()
        if not self.done and now - self._last_log >= self.log_interval_s:
            dt = max(now - self._gen_t0, 1e-9)
            st_desc = self.state
            if self.state == "tool":
                st_desc = f"tool({self.tool_name or '?'}, buf={len(self.tool_tokens)}t)"
            elif self.state == "discard_tool":
                st_desc = f"discard_tool({self.tool_name or '?'}, {self._discard_toks}t)"
            self.log_fn(
                f"[gen] state={st_desc} | {self._n_tok} tok ({self._n_tok / dt:.1f} tok/s) | "
                f"active {mx.get_active_memory() / 1e9:.2f} GB"
            )
            self._last_log = now
        return out
