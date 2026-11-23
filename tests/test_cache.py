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


def test_inplace_move_hit_rollback_on_prefill_abort(monkeypatch):
    """If an in-place (move_hit) prefill fails/aborts mid-way, the original chat cache must be restored."""
    m = _DummyModel()
    pc = ic.SmartPromptCache(max_entries=2, min_tokens=50)
    pc.pin_min_tokens = 20

    def ok_prefill(m, prompt_ids, cache, *, ordered, prefill_chunk, tap_visits=(), drafter=None, pos0=0, output_head=True):
        cache.seq_len += len(prompt_ids)
        return 0, cache, None, pos0

    monkeypatch.setattr(_gen, "_prefill", ok_prefill)
    ids1 = list(range(1, 81))
    m._vision_ctx = {"cache_ids": ids1, "images_plan": [], "pin_hint": 25, "weights_path": None, "encoded_spans": None}
    pc.prefill(m, ids1, ordered=False, prefill_chunk=256)

    chats_before = [e for e in pc._entries if not e.get("pinned")]
    assert len(chats_before) == 1
    assert len(chats_before[0]["tokens"]) == 79
    assert chats_before[0]["cache"].seq_len == 79

    def failing_prefill(m, prompt_ids, cache, *, ordered, prefill_chunk, tap_visits=(), drafter=None, pos0=0, output_head=True):
        cache.seq_len += 10  # partially mutate cache before failing
        raise RuntimeError("simulated Metal OOM or ClientGone")

    monkeypatch.setattr(_gen, "_prefill", failing_prefill)
    ids2 = ids1 + list(range(200, 230))
    m._vision_ctx = {"cache_ids": ids2, "images_plan": [], "pin_hint": 25, "weights_path": None, "encoded_spans": None}
    try:
        pc.prefill(m, ids2, ordered=False, prefill_chunk=256)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    # Chat cache must be rolled back to seq_len=79 and restored into _entries!
    chats_after = [e for e in pc._entries if not e.get("pinned")]
    assert len(chats_after) == 1
    assert len(chats_after[0]["tokens"]) == 79
    assert chats_after[0]["cache"].seq_len == 79


def test_pre_eviction_and_large_chat_protection(monkeypatch):
    """Verify (1) short side queries off pinned-sys do not evict a 4096+ token chat when max_entries=2,
    and (2) a large new chat pre-evicts the old chat BEFORE prefilling."""
    m = _DummyModel()
    pc = ic.SmartPromptCache(max_entries=2, min_tokens=500)
    pc.pin_min_tokens = 1000

    entries_during_prefill: list[int] = []

    def tracking_prefill(m, prompt_ids, cache, *, ordered, prefill_chunk, tap_visits=(), drafter=None, pos0=0, output_head=True):
        if not output_head and pos0 >= 1000:
            entries_during_prefill.append(sum(1 for e in pc._entries if not e.get("pinned")))
        cache.seq_len += len(prompt_ids)
        return 0, cache, None, pos0

    monkeypatch.setattr(_gen, "_prefill", tracking_prefill)

    pin_prefix = list(range(1, 1001))
    # Main chat: 1000 pinned + 5000 unpinned = 6000 tokens
    main_chat = pin_prefix + list(range(2000, 7000))
    m._vision_ctx = {"cache_ids": main_chat, "images_plan": [], "pin_hint": 1000, "weights_path": None, "encoded_spans": None}
    pc.prefill(m, main_chat, ordered=False, prefill_chunk=256)
    chats = [e for e in pc._entries if not e.get("pinned")]
    assert len(chats) == 1 and len(chats[0]["tokens"]) == 5999

    # Short side query: 1000 pinned + 200 unpinned = 1200 tokens (>= min_tokens=500, but +200 < 500 unpinned)
    side_query = pin_prefix + list(range(9000, 9200))
    m._vision_ctx = {"cache_ids": side_query, "images_plan": [], "pin_hint": 1000, "weights_path": None, "encoded_spans": None}
    _, _, _, _, reused_side = pc.prefill(m, side_query, ordered=False, prefill_chunk=256)
    assert reused_side == 1000
    # Main chat (5999 tok) MUST still be preserved in _entries!
    chats = [e for e in pc._entries if not e.get("pinned")]
    assert len(chats) == 1 and len(chats[0]["tokens"]) == 5999

    # Large new chat: 1000 pinned + 2000 unpinned = 3000 tokens -> pre-evicts old chat BEFORE prefilling!
    entries_during_prefill.clear()
    new_large_chat = pin_prefix + list(range(10000, 12000))
    m._vision_ctx = {"cache_ids": new_large_chat, "images_plan": [], "pin_hint": 1000, "weights_path": None, "encoded_spans": None}
    pc.prefill(m, new_large_chat, ordered=False, prefill_chunk=256)
    assert entries_during_prefill == [0]  # old chat was evicted BEFORE new prefill ran!
    chats = [e for e in pc._entries if not e.get("pinned")]
    assert len(chats) == 1 and len(chats[0]["tokens"]) == 2999


def test_post_gen_kv_promotion_pre_tool_fallback_and_ngram_reuse(monkeypatch):
    """Verify full-turn post-gen KV cache promotion, thinking-end pre_tool_gen fallback, and O(delta) N-gram reuse."""
    m = _DummyModel()
    pc = ic.SmartPromptCache(max_entries=2, min_tokens=50)
    pc.pin_min_tokens = 20

    def fake_prefill(m, prompt_ids, cache, *, ordered, prefill_chunk, tap_visits=(), drafter=None, pos0=0, output_head=True):
        cache.seq_len += len(prompt_ids)
        return 0, cache, None, pos0

    monkeypatch.setattr(_gen, "_prefill", fake_prefill)

    # Turn 1: prompt of 80 tokens
    ids1 = list(range(1, 81))
    m._vision_ctx = {"cache_ids": ids1, "images_plan": [], "pin_hint": 25, "weights_path": None, "encoded_spans": None}
    _, live_cache, _, _, reused1 = pc.prefill(m, ids1, ordered=False, prefill_chunk=256)
    assert reused1 == 0
    assert live_cache.seq_len == 80

    # Simulate stream_dspark generating 40 thinking tokens + 20 tool-call JSON tokens (60 committed + 1 anchor)
    ng = _gen.NgramDraftIndex(ids1)
    thinking_toks = list(range(1000, 1040))
    ng.extend(thinking_toks + [200049])  # 40 committed thinking tokens, 200049 is current anchor
    live_cache.seq_len = 80 + 40
    pc.checkpoint_pre_tool_gen(ng)

    tool_json_toks = list(range(2000, 2020))
    ng.extend(tool_json_toks)  # 20 more committed tokens, last is current anchor
    live_cache.seq_len = 80 + 40 + 20
    pc.finalize_post_gen(ng)

    # Incremental N-gram index reuse in O(delta)
    turn2_ids = ids1 + thinking_toks + [200049] + tool_json_toks[:-1] + list(range(3000, 3015))
    reused_ng = pc.acquire_ngram_index(turn2_ids)
    assert reused_ng is ng
    assert reused_ng.tokens == turn2_ids

    # Turn 2A: exact match on post_gen (80 prompt + 40 thinking + 20 tool_json = 140 cached tokens!)
    m._vision_ctx = {"cache_ids": turn2_ids, "images_plan": [], "pin_hint": 25, "weights_path": None, "encoded_spans": None}
    _, live_cache2, _, _, reused2 = pc.prefill(m, turn2_ids, ordered=False, prefill_chunk=256)
    assert reused2 == 140  # Reused ALL 60 generated tokens!

    # Now test pre_tool_gen fallback when tool JSON keys get re-ordered on next turn
    ng2 = _gen.NgramDraftIndex(turn2_ids)
    think2 = list(range(4000, 4050))
    ng2.extend(think2 + [200049])
    live_cache2.seq_len = len(turn2_ids) + 50
    pc.checkpoint_pre_tool_gen(ng2)

    raw_json2 = list(range(5000, 5025))
    ng2.extend(raw_json2)
    live_cache2.seq_len = len(turn2_ids) + 50 + 25
    pc.finalize_post_gen(ng2)

    # Next turn re-orders JSON keys after thinking (so post_gen diverges, but pre_tool_gen matches all 50 thinking tokens!)
    reordered_json2 = list(range(6000, 6030))
    turn3_ids = turn2_ids + think2 + reordered_json2
    m._vision_ctx = {"cache_ids": turn3_ids, "images_plan": [], "pin_hint": 25, "weights_path": None, "encoded_spans": None}
    _, _, _, _, reused3 = pc.prefill(m, turn3_ids, ordered=False, prefill_chunk=256)
    assert reused3 == len(turn2_ids) + 50  # Reused 100% of the 50-token thinking block!


def test_trim_sliding_slots_on_clone_for_wide_prefill_chunks():
    """Verify `_clone_cache_for_store` trims >1024 sliding KV buffers from 512/1024 chunks back to W=1024."""
    import mlx.core as mx
    from runtime_mlx.inkling import model as _model_mod

    cfg = _model_mod.Inkling2Config()
    cache = _model_mod.Inkling2Cache(cfg)
    cache.kv_is_sliding[0] = True
    cache.kv_f16[0] = mx.ones((2, 8, 2048, 128), dtype=mx.float16)
    cache.kv_len[0] = 2048
    cache.kv_cap[0] = 2048
    cache.kv_start[0] = 1000
    mx.eval(cache.kv_f16[0])

    for share_global in (True, False):
        cloned = ic._clone_cache_for_store(cache, share_global=share_global)
        assert cloned.kv_len[0] == 1024
        assert cloned.kv_cap[0] == 1024
        assert cloned.kv_start[0] == 1000 + 1024
        assert cloned.kv_f16[0].shape == (2, 8, 1024, 128)


