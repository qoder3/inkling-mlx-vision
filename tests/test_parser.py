"""Unit tests for MonitoredOutputParser: repetition loop guard & stuck discard_tool guard."""

from __future__ import annotations

from inkling_mlx_vision.parser import MonitoredOutputParser


class _DummyTok:
    def decode(self, toks, **_kw):
        return "".join(chr(65 + (t % 26)) for t in toks)


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
