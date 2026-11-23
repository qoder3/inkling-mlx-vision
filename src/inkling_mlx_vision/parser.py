"""Monitored OutputParser with live token/state logging, FP8 tool-JSON typo repair, and runaway/agent loop guards."""

from __future__ import annotations

import collections
from difflib import SequenceMatcher
import json
import re
import threading
import time
import uuid
from typing import Any, Callable

import mlx.core as mx
from runtime_mlx import server as _server

_OrigOutputParser = _server.OutputParser
_TOOL_JSON_TYPO_RE = re.compile(r'^(\s*\{\s*"name"\s*:\s*"[^"]+")\s*,?\s*"?(args|arguments)"\s*:')
_LOOP_GUARD_TLS = threading.local()


def set_active_loop_guard(info: dict[str, Any] | None) -> None:
    """Bind per-request agent tool-loop circuit-breaker state for the current worker thread."""
    _LOOP_GUARD_TLS.info = info


def get_active_loop_guard() -> dict[str, Any] | None:
    return getattr(_LOOP_GUARD_TLS, "info", None)


def repair_tool_json(raw: str) -> str:
    """Repair Inkling-2 quantized tool JSON missing `,"` before `args`/`arguments`."""
    return _TOOL_JSON_TYPO_RE.sub(r'\1,"\2":', raw, count=1)


def _normalize_arg_val(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.split())
    if isinstance(v, dict):
        return {str(k): _normalize_arg_val(val) for k, val in sorted(v.items())}
    if isinstance(v, list):
        return [_normalize_arg_val(x) for x in v]
    return v


def canonical_tool_args(args: Any) -> str:
    """Return a whitespace- and key-order-normalized canonical JSON string for tool arguments."""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except (ValueError, RecursionError):
            args = {"input": args}
    norm = _normalize_arg_val(args)
    try:
        return json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(norm)


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def _outputs_stagnant(a: str, b: str, *, min_chars: int = 120) -> bool:
    """True when two non-trivial tool outputs (>= min_chars) are identical or >= 90% identical."""
    na = " ".join(a.split())
    nb = " ".join(b.split())
    if len(na) < min_chars or len(nb) < min_chars:
        return False
    if na == nb:
        return True
    # Cap comparison prefix/suffix so huge 12K tool outputs stay O(1) fast (<0.5 ms).
    sa = na[:1024] + na[-1024:] if len(na) > 2048 else na
    sb = nb[:1024] + nb[-1024:] if len(nb) > 2048 else nb
    if abs(len(na) - len(nb)) > max(len(na), len(nb)) * 0.15:
        return False
    return SequenceMatcher(None, sa, sb).quick_ratio() >= 0.90


def detect_agent_tool_loop(
    messages: list[dict[str, Any]],
    *,
    soft_exact: int = 2,
    hard_exact: int = 3,
    soft_stagnant: int = 3,
    hard_stagnant: int = 5,
) -> dict[str, Any] | None:
    """Detect repeated identical or stagnant multi-turn tool-call loops at the tail of `messages`.

    Returns `None` for normal multi-step tool use. Only triggers when `messages[-1]` is a `role: "tool"`
    result following consecutive identical `(tool_name, args)` calls or consecutive same-tool calls
    producing identical non-trivial (>=120 char) outputs.
    """
    if not messages or messages[-1].get("role") != "tool":
        return None

    # Collect trailing (tool_name, canonical_args, tool_output) tuples backwards until a user/plain-assistant boundary.
    history: list[tuple[str, str, str]] = []
    i = len(messages) - 1
    while i >= 1 and len(history) < 12:
        if messages[i].get("role") != "tool":
            break
        # Group consecutive tool results belonging to the preceding assistant tool_calls turn.
        tool_msgs: list[dict[str, Any]] = []
        while i >= 0 and messages[i].get("role") == "tool":
            tool_msgs.append(messages[i])
            i -= 1
        if i < 0 or messages[i].get("role") != "assistant":
            break
        tcs = messages[i].get("tool_calls") or []
        if not tcs:
            break
        # Match the last tool call / tool result of this turn.
        tc = tcs[-1]
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            break
        tname = fn["name"]
        cargs = canonical_tool_args(fn.get("arguments"))
        tout = _extract_text_content(tool_msgs[0].get("content"))
        history.append((tname, cargs, tout))
        i -= 1

    if len(history) < min(soft_exact, soft_stagnant):
        return None

    last_name, last_args, last_out = history[0]
    exact_streak = 1
    for name, cargs, _ in history[1:]:
        if name == last_name and cargs == last_args:
            exact_streak += 1
        else:
            break

    stagnant_streak = 1
    for name, _, tout in history[1:]:
        if name == last_name and _outputs_stagnant(last_out, tout):
            stagnant_streak += 1
        else:
            break

    if exact_streak < soft_exact and stagnant_streak < soft_stagnant:
        return None

    hard_block_sig = (last_name, last_args) if exact_streak >= hard_exact else None
    hard_block_tool = last_name if stagnant_streak >= hard_stagnant else None

    if hard_block_sig is not None or hard_block_tool is not None:
        reason = (
            f"you called `{last_name}` with identical arguments {exact_streak} times in a row"
            if hard_block_sig is not None
            else f"`{last_name}` returned the same output {stagnant_streak} times in a row"
        )
        reminder = (
            f"<system-reminder>CRITICAL LOOP GUARD: {reason}. "
            f"Do NOT repeat this `{last_name}` call again. Stop retrying, change your approach completely, "
            f"or respond directly to the user in plain text explaining what is blocking progress.</system-reminder>"
        )
    else:
        reason = (
            f"you already called `{last_name}` with the exact same arguments {exact_streak} times in a row"
            if exact_streak >= soft_exact
            else f"`{last_name}` returned nearly identical output {stagnant_streak} times in a row"
        )
        reminder = (
            f"<system-reminder>Loop warning: {reason}. "
            f"Do NOT repeat the same call. Try a meaningfully different command/tool, "
            f"or explain the current result to the user.</system-reminder>"
        )

    return {
        "tool_name": last_name,
        "canonical_args": last_args,
        "exact_streak": exact_streak,
        "stagnant_streak": stagnant_streak,
        "hard_block_sig": hard_block_sig,
        "hard_block_tool": hard_block_tool,
        "reminder": reminder,
    }


_TOOL_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"bash", "runcommand", "executecommand", "shell", "terminal"}),
    frozenset({"read", "readfile", "viewfile", "openfile"}),
    frozenset({"write", "writefile", "createfile", "savefile"}),
    frozenset({"edit", "editfile", "replaceinfile", "modifyfile", "strreplace"}),
    frozenset({"grep", "searchfiles", "ripgrep", "findinfiles"}),
    frozenset({"glob", "findfiles", "listfiles"}),
)


def resolve_tool_name(raw_name: Any, tool_names: set[str] | frozenset[str]) -> str | None:
    """Resolve `raw_name` against declared `tool_names` (exact, case-insensitive, snake_case, or unambiguous alias)."""
    if not isinstance(raw_name, str) or not tool_names:
        return None
    s = raw_name.strip()
    if not s:
        return None
    if s in tool_names:
        return s
    s_low = s.lower()
    ci = [t for t in tool_names if t.lower() == s_low]
    if len(ci) == 1:
        return ci[0]

    def _norm(x: str) -> str:
        return x.lower().replace("_", "").replace("-", "")

    s_norm = _norm(s)
    ni = [t for t in tool_names if _norm(t) == s_norm]
    if len(ni) == 1:
        return ni[0]
    for grp in _TOOL_ALIAS_GROUPS:
        if s_norm in grp:
            matches = [t for t in tool_names if _norm(t) in grp]
            if len(matches) == 1:
                return matches[0]
    return None


def inject_tool_loop_reminder(messages: list[dict[str, Any]], loop_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Append a loop-breaker `<system-reminder>` to `messages[-1]` ONLY (preserving 100% of earlier KV cache)."""
    if not loop_info or not messages or messages[-1].get("role") != "tool":
        return messages
    out = list(messages)
    last = dict(out[-1])
    rem = loop_info["reminder"]
    content = last.get("content")
    if isinstance(content, list):
        last["content"] = list(content) + [{"type": "text", "text": "\n\n" + rem}]
    else:
        base = "" if content is None else str(content)
        last["content"] = (base + "\n\n" + rem) if base else rem
    out[-1] = last
    return out


class MonitoredOutputParser(_OrigOutputParser):
    """Extends `lamq serve`'s `OutputParser` with live visibility, FP8 tool-JSON repair, and loop protection."""

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
        # Start timer lazily on first generated token so prefill time never skews live tok/s.
        self._gen_t0: float | None = None
        self._last_log: float | None = None
        self._n_tok = 0
        self._discard_toks = 0
        self._recent: collections.deque[int] = collections.deque(maxlen=self.loop_window)

    def _finish_tool(self, out: list) -> None:
        raw = self._dec(self.tool_tokens)
        repaired = False
        try:
            obj = json.loads(raw, parse_constant=_server._reject_json_constant)
        except ValueError:
            fixed = repair_tool_json(raw)
            if fixed == raw:
                return
            try:
                obj = json.loads(fixed, parse_constant=_server._reject_json_constant)
                repaired = True
            except (ValueError, RecursionError):
                return
        except RecursionError:
            return
        try:
            if not isinstance(obj, dict):
                return
            hdr_resolved = resolve_tool_name(self.tool_name, self.tool_names) if self.tool_name else None
            if not self.tool_name or ("name" in obj and ("args" in obj or "arguments" in obj)):
                obj_raw = obj.get("name")
                obj_resolved = resolve_tool_name(obj_raw, self.tool_names)
                if self.tool_name and obj_raw is not None:
                    if (hdr_resolved or self.tool_name) != (obj_resolved or obj_raw):
                        return
                name = obj_resolved or hdr_resolved
                args = obj.get("args", obj.get("arguments"))
            else:
                name = hdr_resolved
                args = obj.get("arguments", obj)
            if not isinstance(name, str) or name not in self.tool_names or not isinstance(args, dict):
                return
            arguments = json.dumps(args, ensure_ascii=False, allow_nan=False)
            arguments.encode("utf-8")
        except (ValueError, RecursionError):
            return
        if repaired:
            self.log_fn(f"[gen] repaired tool JSON typo for {name!r}")

        if self.loop_guard_enabled:
            guard = get_active_loop_guard()
            if guard is not None:
                sig = (name, canonical_tool_args(args))
                if guard.get("hard_block_sig") == sig or guard.get("hard_block_tool") == name:
                    streak = max(int(guard.get("exact_streak", 0)), int(guard.get("stagnant_streak", 0)))
                    self.log_fn(
                        f"[gen] guard: blocked repeated tool call {name!r} "
                        f"(exact={guard.get('exact_streak')}, stagnant={guard.get('stagnant_streak')})"
                    )
                    out.append(
                        (
                            "content",
                            f"Stopped repeating the same `{name}` tool call after {streak} consecutive "
                            f"identical/stagnant attempts. Please check the previous `{name}` output "
                            f"or tell me how you'd like to proceed.",
                        )
                    )
                    self.done, self.finish = True, "stop"
                    return

        out.append(
            (
                "tool",
                {
                    "id": "call_" + uuid.uuid4().hex[:24],
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                },
            )
        )
        self.done, self.finish = True, "tool_calls"

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
        now = time.perf_counter()
        if self._gen_t0 is None:
            self._gen_t0 = now
            self._last_log = now

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

        if not self.done and self._last_log is not None and now - self._last_log >= self.log_interval_s:
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
