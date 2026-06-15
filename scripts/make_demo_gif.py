"""Assemble a demo GIF from a run's captured per-step screenshots.

No new run / no API cost — it reuses the PNGs written under
results/screenshots/<task_id>/ when an eval was run with --capture-screenshots.
Each frame is captioned with the task goal + step number.

    python -m scripts.make_demo_gif                      # auto-pick the longest run
    python -m scripts.make_demo_gif --task m2w-hard-wiki-compare-pop
"""

from __future__ import annotations

import argparse
import glob
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "results" / "screenshots"


def _goals() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in glob.glob(str(ROOT / "eval" / "tasks" / "*.json")):
        try:
            data = json.loads(Path(f).read_text())
        except Exception:
            continue
        for t in (data.get("tasks", data) if isinstance(data, dict) else data):
            if isinstance(t, dict) and t.get("task_id"):
                out[t["task_id"]] = t.get("goal", "")
    return out


def _auto_task() -> str:
    dirs = [(len(list(d.glob("*.png"))), d.name) for d in SHOTS.iterdir() if d.is_dir()]
    dirs = [x for x in dirs if x[0] >= 3]
    if not dirs:
        raise SystemExit("No screenshots found. Run an eval with --capture-screenshots first.")
    return max(dirs)[1]


def build(task: str, out: Path, width: int, ms: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    frames_dir = SHOTS / task
    pngs = sorted(frames_dir.glob("step*.png"))
    if not pngs:
        raise SystemExit(f"No frames in {frames_dir}")
    goal = _goals().get(task, task)
    try:
        font = ImageFont.load_default(size=15)
    except Exception:
        font = ImageFont.load_default()

    frames = []
    for i, p in enumerate(pngs):
        img = Image.open(p).convert("RGB")
        scale = width / img.width
        img = img.resize((width, int(img.height * scale)))
        # Caption bar on top: goal (wrapped) + step counter.
        wrapped = textwrap.wrap(f"GOAL: {goal}", width=max(20, width // 9))[:2]
        bar_h = 22 * (len(wrapped) + 1) + 10
        canvas = Image.new("RGB", (width, img.height + bar_h), (24, 24, 28))
        d = ImageDraw.Draw(canvas)
        y = 6
        for line in wrapped:
            d.text((10, y), line, fill=(235, 235, 240), font=font)
            y += 22
        d.text((10, y), f"step {i + 1}/{len(pngs)}  ·  web-agent (Sonnet 4.6)",
               fill=(120, 200, 255), font=font)
        canvas.paste(img, (0, bar_h))
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))

    out.parent.mkdir(parents=True, exist_ok=True)
    # Hold the last frame longer so the result is readable.
    durations = [ms] * (len(frames) - 1) + [ms * 2]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({len(frames)} frames, {kb:.0f} KB)  task={task!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a demo GIF from captured screenshots.")
    p.add_argument("--task", default=None, help="task_id under results/screenshots/ (default: longest)")
    p.add_argument("--out", default=str(ROOT / "results" / "reports" / "demo.gif"))
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--ms", type=int, default=1400, help="ms per frame")
    args = p.parse_args(argv)
    build(args.task or _auto_task(), Path(args.out), args.width, args.ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
