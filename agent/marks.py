"""Set-of-Marks (SoM) rendering — overlay numbered boxes on a screenshot.

Production web agents are multimodal: they see the rendered page with each
interactable element boxed and numbered, and address marks by number. This draws
those marks (matching the ``@e`` refs from the observation) onto the viewport
screenshot, so a vision model can use spatial layout the a11y tree can't convey.

Reference: Yang et al., "Set-of-Mark Prompting…" — the standard technique behind
strong GPT-4V / computer-use web-agent results.
"""

from __future__ import annotations

import base64
import io
from typing import Any

# A small palette so adjacent boxes are distinguishable.
_COLORS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
]


def render_marked_screenshot(
    png_bytes: bytes,
    elements: list[dict[str, Any]],
    *,
    max_marks: int = 60,
) -> str:
    """Draw numbered boxes for on-screen elements; return base64 PNG.

    Only elements whose rect falls within the screenshot are marked (offscreen
    elements stay in the text list but aren't drawn). The label is the ref's
    number, so ``@e12`` is box ``12``.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = ImageFont.load_default()

    drawn = 0
    for e in elements:
        if drawn >= max_marks:
            break
        rect = e.get("rect") or {}
        x, y = rect.get("x"), rect.get("y")
        w, h = rect.get("w", 0), rect.get("h", 0)
        if x is None or y is None or w <= 1 or h <= 1:
            continue
        # Skip elements outside the captured viewport.
        if x + w < 0 or y + h < 0 or x > W or y > H:
            continue
        num = e["ref"].lstrip("@e") or "?"
        color = _COLORS[drawn % len(_COLORS)]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W - 1, x + w), min(H - 1, y + h)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        # Label chip at the top-left corner of the box.
        label = num
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = y0 - th - 3 if y0 - th - 3 >= 0 else y0
        draw.rectangle([x0, ly, x0 + tw + 4, ly + th + 4], fill=color)
        draw.text((x0 + 2, ly + 1), label, fill=(255, 255, 255), font=font)
        drawn += 1

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
