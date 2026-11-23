"""Unit tests for MonitoredOutputParser: repetition loop guard & stuck discard_tool guard."""

from __future__ import annotations

from inkling_mlx_vision.parser import (
    MonitoredOutputParser,
    detect_agent_tool_loop,
    inject_tool_loop_reminder,
    repair_tool_json,
    resolve_tool_name,
    set_active_loop_guard,
)


class _DummyTok:
    def decode(self, toks, **_kw):
        return "".join(chr(65 + (t % 26)) for t in toks)


class _AsciiTok:
    def decode(self, toks, **_kw):
        return "".join(chr(t) for t in toks)


_IDS = {
    "user": 200000,
    "model": 200001,
    "system": 200002,
    "tool": 200003,
    "text": 200004,
    "thinking": 200005,
    "eos": 200006,
    "end": 200010,
    "tool_json": 200011,
    "tool_text": 200012,
}


def test_no_false_positive_on_varied_output():
    p = MonitoredOutputParser(_DummyTok(), _IDS, tool_names={"Read"})
    p.feed(_IDS["text"])
    for i in range(400):
        p.feed(10 + (i * 17) % 1000)
    assert not p.done


def test_catches_verbatim_repetition_loop():
    p = MonitoredOutputParser(_DummyTok(), _IDS, tool_names={"Read"})
    p.feed(_IDS["text"])
    for i in range(350):
        p.feed(50 + (i % 8))
        if p.done:
            break
    assert p.done


def test_catches_stuck_discard_tool():
    p = MonitoredOutputParser(_DummyTok(), _IDS, tool_names={"Read"})
    p.max_discard_tool_tokens = 64
    p.feed(_IDS["model"])
    p.feed(1)  # undeclared tool name
    p.feed(_IDS["tool_text"])  # enters discard_tool
    assert p.state == "discard_tool"
    for i in range(100):
        p.feed(100 + i)
        if p.done:
            break
    assert p.done


def test_repair_fp8_tool_json_typo():
    assert repair_tool_json('{"name":"read_file"args":{"path":"a.py"}}') == '{"name":"read_file","args":{"path":"a.py"}}'
    assert repair_tool_json('{"name": "read_file" arguments": {"path": "a.py"}}') == '{"name": "read_file","arguments": {"path": "a.py"}}'
    assert repair_tool_json('{"name":"read_file""args":{"path":"a.py"}}') == '{"name":"read_file","args":{"path":"a.py"}}'

    logs: list[str] = []
    p = MonitoredOutputParser(_AsciiTok(), _IDS, tool_names={"read_file"})
    p.log_fn = logs.append
    assert p._gen_t0 is None

    events = []
    events.extend(p.feed(_IDS["model"]))
    assert p._gen_t0 is not None
    events.extend(p.feed(_IDS["tool_json"]))
    for ch in '{"name":"read_file"args":{"path":"a.py"}}':
        events.extend(p.feed(ord(ch)))
    events.extend(p.feed(_IDS["end"]))

    assert p.done and p.finish == "tool_calls"
    assert len(events) == 1 and events[0][0] == "tool"
    assert events[0][1]["function"]["name"] == "read_file"
    assert events[0][1]["function"]["arguments"] == '{"path": "a.py"}'
    assert any("repaired tool JSON typo" in msg for msg in logs)


def _tool_turn(name: str, args: dict, output: str | list) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}],
        },
        {"role": "tool", "name": name, "tool_call_id": "c1", "content": output},
    ]


def test_agent_tool_loop_no_false_positive_on_distinct_commands():
    msgs = [{"role": "user", "content": "create dirs"}]
    msgs += _tool_turn("Bash", {"command": "mkdir a"}, "OK")
    msgs += _tool_turn("Bash", {"command": "mkdir b"}, "OK")
    msgs += _tool_turn("Bash", {"command": "mkdir c"}, "OK")
    assert detect_agent_tool_loop(msgs) is None


def test_agent_tool_loop_exact_soft_and_hard_breaker():
    msgs = [{"role": "user", "content": "fix tests"}]
    msgs += _tool_turn("Bash", {"command": "pytest tests/a.py"}, "AssertionError: 1 != 2")
    assert detect_agent_tool_loop(msgs) is None

    # 2nd identical call (different whitespace) -> soft reminder only, KV prefix messages[:-1] untouched
    msgs += _tool_turn("Bash", {"command": "pytest   tests/a.py"}, "AssertionError: 1 != 2")
    info2 = detect_agent_tool_loop(msgs)
    assert info2 is not None
    assert info2["exact_streak"] == 2
    assert info2["hard_block_sig"] is None
    patched = inject_tool_loop_reminder(msgs, info2)
    assert patched[:-1] == msgs[:-1]
    assert "<system-reminder>" in patched[-1]["content"]

    # 3rd identical call -> hard circuit breaker armed
    msgs += _tool_turn("Bash", {"command": "pytest tests/a.py"}, "AssertionError: 1 != 2")
    info3 = detect_agent_tool_loop(msgs)
    assert info3 is not None
    assert info3["exact_streak"] == 3
    assert info3["hard_block_sig"] is not None

    # Parser blocks 4th identical Bash call cleanly, turning it into plain-text stop
    logs: list[str] = []
    p = MonitoredOutputParser(_AsciiTok(), _IDS, tool_names={"Bash"})
    p.log_fn = logs.append
    set_active_loop_guard(info3)
    try:
        ev = []
        ev.extend(p.feed(_IDS["model"]))
        ev.extend(p.feed(_IDS["tool_json"]))
        for ch in '{"name":"Bash","args":{"command":"pytest tests/a.py"}}':
            ev.extend(p.feed(ord(ch)))
        ev.extend(p.feed(_IDS["end"]))
    finally:
        set_active_loop_guard(None)

    assert p.done and p.finish == "stop"
    assert len(ev) == 1 and ev[0][0] == "content"
    assert "Bash" in ev[0][1]
    assert any("blocked repeated tool call" in m for m in logs)

    # A DIFFERENT Bash command is STILL allowed even while the repeated signature is blocked!
    p2 = MonitoredOutputParser(_AsciiTok(), _IDS, tool_names={"Bash"})
    set_active_loop_guard(info3)
    try:
        ev2 = []
        ev2.extend(p2.feed(_IDS["model"]))
        ev2.extend(p2.feed(_IDS["tool_json"]))
        for ch in '{"name":"Bash","args":{"command":"git status"}}':
            ev2.extend(p2.feed(ord(ch)))
        ev2.extend(p2.feed(_IDS["end"]))
    finally:
        set_active_loop_guard(None)

    assert p2.done and p2.finish == "tool_calls"
    assert ev2[0][0] == "tool" and ev2[0][1]["function"]["name"] == "Bash"


def test_agent_tool_loop_stagnant_large_output_and_multimodal_tool():
    big_err = "Traceback (most recent call last):\n" + ("  File 'app.py', line 42, in run\n" * 8)
    msgs = [{"role": "user", "content": "debug"}]
    msgs += _tool_turn("Bash", {"command": "python app.py --v1"}, big_err)
    msgs += _tool_turn("Bash", {"command": "python app.py --v2"}, big_err)
    msgs += _tool_turn("Bash", {"command": "python app.py --v3"}, [{"type": "text", "text": big_err}, {"type": "image"}])
    info = detect_agent_tool_loop(msgs)
    assert info is not None
    assert info["stagnant_streak"] == 3
    assert info["hard_block_tool"] is None
    patched = inject_tool_loop_reminder(msgs, info)
    assert isinstance(patched[-1]["content"], list)
    assert "<system-reminder>" in patched[-1]["content"][-1]["text"]


def test_fuzzy_tool_name_resolution_in_parser():
    tools = {"Bash", "Read", "Write", "Edit", "TodoWrite"}
    assert resolve_tool_name("bash", tools) == "Bash"
    assert resolve_tool_name("run_command", tools) == "Bash"
    assert resolve_tool_name("read_file", tools) == "Read"
    assert resolve_tool_name("write_file", tools) == "Write"
    assert resolve_tool_name("edit_file", tools) == "Edit"
    assert resolve_tool_name("todo_write", tools) == "TodoWrite"
    assert resolve_tool_name("unknown_tool", tools) is None

    # Parser accepts header "bash" + JSON "run_command" and emits canonical "Bash"
    p = MonitoredOutputParser(_AsciiTok(), _IDS, tool_names=tools)
    ev = []
    ev.extend(p.feed(_IDS["model"]))
    for ch in "bash":
        ev.extend(p.feed(ord(ch)))
    ev.extend(p.feed(_IDS["tool_json"]))
    for ch in '{"name":"run_command","args":{"command":"ls"}}':
        ev.extend(p.feed(ord(ch)))
    ev.extend(p.feed(_IDS["end"]))
    assert p.done and p.finish == "tool_calls"
    assert len(ev) == 1 and ev[0][0] == "tool"
    assert ev[0][1]["function"]["name"] == "Bash"
    assert ev[0][1]["function"]["arguments"] == '{"command": "ls"}'

