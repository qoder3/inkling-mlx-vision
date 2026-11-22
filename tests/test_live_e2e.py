"""Optional live end-to-end multimodal test against a running server.

Run manually when the server is listening on http://127.0.0.1:8080:
    INKLING_LIVE_URL=http://127.0.0.1:8080 pytest tests/test_live_e2e.py -s
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

import pytest
from PIL import Image, ImageDraw, ImageFont

LIVE_URL = os.environ.get("INKLING_LIVE_URL")


def _get_font(size: int):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _make_card(bg, shape, shape_color, label) -> str:
    im = Image.new("RGB", (400, 280), bg)
    d = ImageDraw.Draw(im)
    if shape == "circle":
        d.ellipse([130, 30, 270, 170], fill=shape_color)
    elif shape == "square":
        d.rectangle([130, 30, 270, 170], fill=shape_color)
    elif shape == "triangle":
        d.polygon([(200, 30), (120, 170), (280, 170)], fill=shape_color)
    d.text((70, 200), label, fill=(0, 0, 0), font=_get_font(48))
    return _data_uri(im)


def _post_chat(messages, *, max_tokens=192):
    body = json.dumps(
        {
            "model": "inkling-2-mini",
            "messages": messages,
            "stream": False,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"{LIVE_URL.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


@pytest.mark.skipif(not LIVE_URL, reason="Set INKLING_LIVE_URL=http://127.0.0.1:8080 to run live E2E tests")
def test_multi_image_and_tool_image_e2e():
    img_a = _make_card((255, 255, 255), "circle", (220, 30, 30), "ALPHA 11")
    img_b = _make_card((255, 255, 255), "square", (30, 90, 230), "BETA 22")

    # 1. Two images in a single user message
    r1 = _post_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_a}},
                    {"type": "image_url", "image_url": {"url": img_b}},
                    {"type": "text", "text": "Read the exact label on the first image and the second image."},
                ],
            }
        ]
    )
    txt1 = r1["choices"][0]["message"]["content"]
    assert "ALPHA" in txt1.upper() and "11" in txt1
    assert "BETA" in txt1.upper() and "22" in txt1

    # 2. Image returned inside a `role: "tool"` message
    img_t = _make_card((255, 255, 255), "triangle", (20, 175, 50), "TOOL 77")
    r2 = _post_chat(
        [
            {"role": "user", "content": "Take a screenshot and read the text."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_shot", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_shot",
                "name": "screenshot",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_t}},
                    {"type": "text", "text": "Screenshot captured."},
                ],
            },
        ]
    )
    txt2 = r2["choices"][0]["message"]["content"]
    assert "TOOL" in txt2.upper() and "77" in txt2
