"""Generate the report charts + tables from trajectory JSONL files.

Reads every run under results/trajectories/ (one JSONL per run), then renders:
  1. cost_vs_success.png        — cost-vs-success Pareto (per run)
  2. sandbox_vs_realistic.png   — WebArena SR vs Mind2Web SR (the punchline)
  3. reflection_ablation.png    — reflection ON vs OFF
  4. failure_taxonomy.png       — failure category breakdown
  5. summary.csv / summary.md   — the run-level table

Run filenames are expected to look like:
    <base>__<model>__reflect-on.jsonl   (as produced by eval.harness)
but the script also infers what it can if they don't.

    python -m scripts.make_charts
    python -m scripts.make_charts --results-dir results/trajectories --out-dir results/reports
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from eval import failure_taxonomy, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _parse_filename(path: Path) -> dict[str, str]:
    stem = path.stem
    parts = stem.split("__")
    base = parts[0] if parts else stem
    model = parts[1] if len(parts) > 1 else "unknown"
    reflect = "on" if (len(parts) > 2 and "on" in parts[2]) else "off"
    return {"label": stem, "base": base, "model": model, "reflect": reflect}


def _benchmark_of(trajs: list[dict[str, Any]]) -> str:
    cnt: dict[str, int] = {}
    for t in trajs:
        b = (t.get("score_detail", {}) or {}).get("benchmark")
        if not b:
            tid = t.get("task_id", "")
            b = tid.split("-", 1)[0] if "-" in tid else "custom"
        cnt[b] = cnt.get(b, 0) + 1
    return max(cnt, key=cnt.get) if cnt else "custom"


def load_runs(results_dir: Path, pattern: str = "*.jsonl") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fp in sorted(glob.glob(str(results_dir / pattern))):
        path = Path(fp)
        trajs = metrics.load_trajectories(path)
        if not trajs:
            continue
        meta = _parse_filename(path)
        summ = metrics.summarize(trajs)
        rows.append(
            {
                **meta,
                "benchmark": _benchmark_of(trajs),
                "n": summ["n_scored"],
                "success_rate": summ["success_rate"],
                "mean_cost_usd": summ["mean_cost_usd"],
                "mean_steps": summ["mean_steps"],
                "mean_latency_s": summ["mean_latency_s"],
                "step_efficiency": summ["step_efficiency"],
                "vision_fallback_rate": summ["vision_fallback_rate"],
                "_trajs": trajs,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart_cost_vs_success(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["mean_cost_usd"], df["success_rate"], s=90, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(
            f"{r['model']}\n({r['reflect']})",
            (r["mean_cost_usd"], r["success_rate"]),
            textcoords="offset points", xytext=(8, 4), fontsize=8,
        )
    ax.set_xlabel("Mean cost per task (USD)")
    ax.set_ylabel("Success rate")
    ax.set_title("Cost vs. success (Pareto)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_sandbox_vs_realistic(df: pd.DataFrame, out: Path) -> None:
    by_bench = df.groupby("benchmark")["success_rate"].max()
    pairs = [(b, by_bench[b]) for b in ("webarena", "mind2web", "local") if b in by_bench]
    if len(pairs) < 1:
        return
    labels = [b for b, _ in pairs]
    vals = [v for _, v in pairs]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=["#2b8cbe", "#e34a33", "#74a9cf"][: len(vals)])
    ax.bar_label(bars, fmt="%.2f")
    ax.set_ylabel("Best success rate")
    ax.set_title("Sandbox vs. realistic")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_reflection_ablation(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["reflect"].isin(["on", "off"])]
    if sub["reflect"].nunique() < 2:
        return
    pivot = sub.pivot_table(
        index="model", columns="reflect", values="success_rate", aggfunc="max"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    pivot.plot(kind="bar", ax=ax, color={"off": "#bdbdbd", "on": "#31a354"})
    ax.set_ylabel("Success rate")
    ax.set_title("Reflection ablation (ON vs OFF)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="reflection")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def chart_failure_taxonomy(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    agg: dict[str, int] = {}
    for trajs in df["_trajs"]:
        for cat, n in failure_taxonomy.breakdown(trajs).items():
            agg[cat] = agg.get(cat, 0) + n
    if not agg:
        return
    cats = sorted(agg, key=agg.get, reverse=True)
    vals = [agg[c] for c in cats]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(cats[::-1], vals[::-1], color="#c994c7")
    ax.bar_label(bars)
    ax.set_xlabel("Count (failed trajectories)")
    ax.set_title("Failure taxonomy")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "label", "benchmark", "model", "reflect", "n", "success_rate",
        "mean_cost_usd", "mean_steps", "step_efficiency", "mean_latency_s",
        "vision_fallback_rate",
    ]
    table = df[cols] if not df.empty else pd.DataFrame(columns=cols)
    table.to_csv(out_dir / "summary.csv", index=False)
    md = table.to_markdown(index=False) if not table.empty else "(no runs found)"
    (out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render report charts from trajectories.")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "trajectories"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "reports"))
    parser.add_argument("--glob", default="*.jsonl")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(Path(args.results_dir), args.glob)

    if df.empty:
        print(f"No trajectory files found in {args.results_dir}. Run the harness first.")
        write_tables(df, out_dir)
        return 0

    chart_cost_vs_success(df, out_dir / "cost_vs_success.png")
    chart_sandbox_vs_realistic(df, out_dir / "sandbox_vs_realistic.png")
    chart_reflection_ablation(df, out_dir / "reflection_ablation.png")
    chart_failure_taxonomy(df, out_dir / "failure_taxonomy.png")
    write_tables(df, out_dir)

    print(f"Wrote charts + tables to {out_dir} ({len(df)} run(s)).")
    for f in sorted(os.listdir(out_dir)):
        print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
