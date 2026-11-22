"""Unit tests for SmartPromptCache: vision-hash safety, ViT skip on hit, dynamic preamble pinning."""

from __future__ import annotations

import numpy as np

from inkling_mlx_vision import cache as ic
from inkling_mlx_vision import vision as iv
from runtime_mlx.inkling import generate as _gen


class _DummyCache:
    clone_count: int = 0

    def __init__(self, seq_len: int = 0):
        self.seq_len = seq_len
        self.state = []
        self._reserve_cap = 0

    def clone(self):
        _DummyCache.clone_count += 1
        return _DummyCache(self.seq_len)


class _DummyModel:
    def __init__(self):
        self._vision_ctx = None
        self.prefilled_ranges: list[tuple[int, int]] = []

    def make_cache(self):
        return _DummyCache(0)

    def forward_prefill_chunk(self, ids, cache=None, tap_visits=()):
        cache.seq_len += len(ids)
        return None, cache

    def forward_argmax(self, ids, cache=None, tap_visits=()):
        cache.seq_len += len(ids)
        return 0, cache

    def forward_ordered_logits(self, ids, cache=None, tap_visits=()):
        cache.seq_len += len(ids)
        return 0, cache


def test_count_preamble_messages():
    msgs = [
        {"role": "system", "content": "sys1"},
        {"role": "system", "content": "sys2"},
        {"role": "user", "content": "<system-reminder>\nEnv info\n</system-reminder>"},
        {"role": "user", "content": "  <system-reminder>\nSkills\n</system-reminder>"},
        {"role": "user", "content": "Fix the bug"},
        {"role": "user", "content": "<system-reminder>\nLate reminder\n</system-reminder>"},
    ]
    assert ic.count_preamble_messages(msgs) == 4


def test_image_cache_ids_no_collision():
    ids_a = ic.make_image_cache_ids("a" * 32, 96)
    ids_b = ic.make_image_cache_ids("b" * 32, 96)
    assert all(x < 0 for x in ids_a)
    assert all(x < 0 for x in ids_b)
    assert len(set(ids_a)) == 96
    assert set(ids_a).isdisjoint(set(ids_b))


def test_smart_prompt_cache_vision_hit_miss_and_dynamic_pin(monkeypatch):
    # Patch _gen._prefill to a lightweight stub and vision.encode_images to count calls
    encode_calls: list[list[str]] = []

    def fake_encode(preps, weights_path):
        encode_calls.append([p.img_hash for p in preps])
        return [np.zeros((p.num_tokens, 4), dtype=np.float32) for p in preps]

    def fake_prefill(m, prompt_ids, cache, *, ordered, prefill_chunk, tap_visits=(), drafter=None, pos0=0, output_head=True):
        ic.ensure_vision_encoded(m, pos0, lambda _: None)
        cache.seq_len += len(prompt_ids)
        return 0, cache, None, pos0

    monkeypatch.setattr(iv, "encode_images", fake_encode)
    monkeypatch.setattr(_gen, "_prefill", fake_prefill)

    m = _DummyModel()
    pc = ic.SmartPromptCache(max_entries=3, min_tokens=50)
    pc.pin_min_tokens = 20

    def prep(h: str, n: int = 10):
        return iv.PreprocessedImage(np.zeros((n, 40, 40, 3), dtype=np.float32), 2, 5, h, (200, 80), (200, 80))

    # Chat 1 with preamble (0..30) + image A (30..40) + text (40..70)
    pre_a = list(range(1, 31))
    img_a = prep("1" * 32, 10)
    ids1 = pre_a + [200054] * 10 + list(range(100, 130))
    cids1 = pre_a + ic.make_image_cache_ids(img_a.img_hash, 10) + list(range(100, 130))
    m._vision_ctx = {"cache_ids": cids1, "images_plan": [(30, 40, img_a)], "pin_hint": 30, "weights_path": "x", "encoded_spans": None}
    _, _, _, _, reused1 = pc.prefill(m, ids1, ordered=False, prefill_chunk=256)
    assert reused1 == 0
    assert encode_calls == [[img_a.img_hash]]

    # Turn 2 of Chat 1 (same image A, appended tokens) -> full HIT 69/80 AND 0 ViT calls!
    ids2 = ids1 + list(range(200, 210))
    cids2 = cids1 + list(range(200, 210))
    m._vision_ctx = {"cache_ids": cids2, "images_plan": [(30, 40, img_a)], "pin_hint": 30, "weights_path": "x", "encoded_spans": None}
    _, _, _, _, reused2 = pc.prefill(m, ids2, ordered=False, prefill_chunk=256)
    assert reused2 == len(ids1) - 1
    assert len(encode_calls) == 1  # ViT skipped!

    # Chat 2 with SAME preamble (0..30) + DIFFERENT image B of SAME 10-token size -> hits pinned preamble (30), NOT Chat 1!
    img_b = prep("2" * 32, 10)
    ids3 = pre_a + [200054] * 10 + list(range(100, 130))
    cids3 = pre_a + ic.make_image_cache_ids(img_b.img_hash, 10) + list(range(100, 130))
    m._vision_ctx = {"cache_ids": cids3, "images_plan": [(30, 40, img_b)], "pin_hint": 30, "weights_path": "x", "encoded_spans": None}
    _, _, _, _, reused3 = pc.prefill(m, ids3, ordered=False, prefill_chunk=256)
    assert reused3 == 30  # hit pinned preamble, zero cross-image collision!
    assert encode_calls[-1] == [img_b.img_hash]

    # Chat 3 after memory/MCP preamble edit -> misses old pin, refreshes pinned slot dynamically!
    pre_v2 = list(range(500, 535))
    ids4 = pre_v2 + list(range(600, 630))
    m._vision_ctx = {"cache_ids": ids4, "images_plan": [], "pin_hint": 35, "weights_path": "x", "encoded_spans": None}
    _, _, _, _, reused4 = pc.prefill(m, ids4, ordered=False, prefill_chunk=256)
    assert reused4 == 0
    pinned = [e for e in pc._entries if e.get("pinned")]
    assert len(pinned) == 1 and pinned[0]["tokens"] == pre_v2

    # Turn 2 of Chat 3 extends non-pinned chat -> should move in-place (only 1 clone at _store, not 2!)
    _DummyCache.clone_count = 0
    reserves_seen: list[int] = []

    def spy_prefill(m, prompt_ids, cache, *, ordered, prefill_chunk, tap_visits=(), drafter=None, pos0=0, output_head=True):
        reserves_seen.append(int(getattr(cache, "_reserve_cap", 0)))
        cache.seq_len += len(prompt_ids)
        return 0, cache, None, pos0

    monkeypatch.setattr(_gen, "_prefill", spy_prefill)
    ids5 = ids4 + list(range(700, 720))
    m._vision_ctx = {"cache_ids": ids5, "images_plan": [], "pin_hint": 35, "weights_path": "x", "encoded_spans": None}
    _, out_cache, _, _, reused5 = pc.prefill(m, ids5, ordered=False, prefill_chunk=256)
    assert reused5 == len(ids4) - 1
    assert _DummyCache.clone_count == 1  # zero-copy move on HIT + 1 clone at _store!
    assert reserves_seen[0] == len(ids5) + 512
    assert out_cache._reserve_cap == 0  # cleanly reset in finally
