"""Unit tests for MonitoredOutputParser: repetition loop guard & stuck discard_tool guard."""

from __future__ import annotations

from inkling_mlx_vision.parser import MonitoredOutputParser, repair_tool_json


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
