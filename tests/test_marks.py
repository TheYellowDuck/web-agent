"""Set-of-Marks renderer."""

import base64
import io

from agent.marks import render_marked_screenshot


def _blank_png(w=200, h=120):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_render_returns_valid_png_same_size():
    from PIL import Image
    els = [
        {"ref": "@e1", "role": "button", "name": "A", "rect": {"x": 10, "y": 10, "w": 40, "h": 20}},
        {"ref": "@e2", "role": "link", "name": "B", "rect": {"x": 60, "y": 50, "w": 30, "h": 15}},
    ]
    out = render_marked_screenshot(_blank_png(), els)
    img = Image.open(io.BytesIO(base64.b64decode(out)))
    assert img.size == (200, 120)
    # something was drawn (no longer all-white)
    assert img.convert("RGB").getextrema() != ((255, 255, 255), (255, 255, 255), (255, 255, 255))


def test_render_skips_missing_or_offscreen_rects():
    # No rect / offscreen / zero-size -> no crash, still a valid PNG.
    els = [
        {"ref": "@e1", "role": "link", "name": "x"},                       # no rect
        {"ref": "@e2", "role": "link", "name": "y", "rect": {"x": 9999, "y": 9999, "w": 5, "h": 5}},
        {"ref": "@e3", "role": "link", "name": "z", "rect": {"x": 1, "y": 1, "w": 0, "h": 0}},
    ]
    out = render_marked_screenshot(_blank_png(), els)
    assert base64.b64decode(out)[:8] == b"\x89PNG\r\n\x1a\n"
