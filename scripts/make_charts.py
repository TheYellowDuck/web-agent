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

from eval import failure_taxonomy, metrics, stats  # noqa: E402

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


def _pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Cost-efficient (non-dominated) points: no other run is both cheaper-or-equal
    and at least as successful."""
    pts = df[["mean_cost_usd", "success_rate"]].to_numpy()
    keep = []
    for i, (c, s) in enumerate(pts):
        dominated = any(
            (pts[j, 0] <= c and pts[j, 1] >= s and (pts[j, 0] < c or pts[j, 1] > s))
            for j in range(len(pts)) if j != i
        )
        if not dominated:
            keep.append(i)
    return df.iloc[keep].sort_values("mean_cost_usd")


def chart_cost_vs_success(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    import itertools

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    benches = sorted(df["benchmark"].unique())
    palette = {"webarena": "#e34a33", "mind2web": "#2b8cbe", "local": "#41ab5d"}
    cyc = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = {b: palette.get(b) or next(cyc) for b in benches}

    for _, r in df.iterrows():
        c = colors[r["benchmark"]]
        on = r["reflect"] == "on"
        ax.scatter(r["mean_cost_usd"], r["success_rate"], s=120, zorder=3, alpha=0.9,
                   facecolor=(c if on else "white"), edgecolor=c, linewidths=1.8)
        m = r["model"].lower()  # label only the model-sweep standouts (not the default Sonnet)
        tier = "Opus" if "opus" in m else ("Haiku" if "haiku" in m else None)
        if tier:
            ax.annotate(tier, (r["mean_cost_usd"], r["success_rate"]),
                        textcoords="offset points", xytext=(8, 5), fontsize=9,
                        fontweight="bold", color=c)

    # Cost-efficient frontier — only draw it if it's non-degenerate (spans a
    # real cost range), so we never show a legend entry for an invisible line.
    fr = _pareto_frontier(df)
    drew_frontier = (
        len(fr) >= 2
        and fr["mean_cost_usd"].nunique() >= 2
        and fr["success_rate"].nunique() >= 2
    )
    if drew_frontier:
        ax.plot(fr["mean_cost_usd"], fr["success_rate"], "--", color="#555",
                lw=1.3, zorder=2)

    ax.set_xlabel("Mean cost per task (USD)")
    ax.set_ylabel("Success rate")
    ax.set_title("Cost vs. success across runs", pad=12)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[b],
                      markeredgecolor=colors[b], markersize=9, label=b) for b in benches]
    handles += [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#555",
               markeredgecolor="#555", markersize=9, label="reflect ON (filled)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="#555", markersize=9, label="reflect OFF (hollow)"),
    ]
    if drew_frontier:
        handles.append(Line2D([0], [0], linestyle="--", color="#555",
                              label="cost-efficient frontier"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)
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


def write_significance(df: pd.DataFrame, out_dir: Path) -> None:
    """Emit significance.md: the correct test for each comparison the runs allow.

    Paired reflect ON-vs-OFF (same base + model) → McNemar; the largest WebArena
    vs the largest Mind2Web run → two-proportion z / Fisher + Newcombe diff CI.
    Makes the statistics a generated artifact, not just a manual CLI call.
    """
    def fmt_p(p: float | None) -> str:
        # Owns the comparator so callers write "p {fmt_p(...)}" → "p < 1e-5" /
        # "p = 1". A rounded 0.0 means "below display resolution", not zero.
        if p is None:
            return "= n/a"
        return "< 1e-5" if p < 1e-5 else f"= {p:g}"

    lines = [
        "# Significance tests",
        "",
        "Auto-generated from `results/trajectories/`. Paired comparisons (same "
        "task set) use McNemar's test; unpaired cross-benchmark comparisons use a "
        "two-proportion z-test + Fisher's exact + a Newcombe difference-of-"
        "proportions CI. Non-overlapping CIs are *not* used as the test.",
        "",
    ]
    wrote = False

    # Paired: reflection ablation, grouped by (base, model).
    for (base, model), g in df.groupby(["base", "model"]):
        refl = {r["reflect"]: r["_trajs"] for _, r in g.iterrows()}
        if "on" in refl and "off" in refl:
            r = stats.compare_paired(refl["on"], refl["off"], label_a="on", label_b="off")
            mc = r["mcnemar"]
            verdict = "**significant**" if mc["significant_05"] else (
                "no detectable effect (note: low power if discordant pairs are few)"
            )
            lines += [
                f"## Reflection ablation — {base} ({model})  ·  paired / McNemar",
                f"- ON {r['on_rate']} vs OFF {r['off_rate']} on {r['n_shared_tasks']} shared tasks",
                f"- discordant pairs: ON-only {r['table']['on_only']}, "
                f"OFF-only {r['table']['off_only']} (total {mc.get('discordant', mc['b'] + mc['c'])})",
                f"- **p {fmt_p(mc['p_value'])}** ({mc['method']}) → {verdict}",
                "",
            ]
            wrote = True

    # Unpaired: best (largest n) WebArena vs best Mind2Web.
    best: dict[str, Any] = {}
    for _, r in df.iterrows():
        b = r["benchmark"]
        if b not in best or r["n"] > best[b]["n"]:
            best[b] = r
    if "webarena" in best and "mind2web" in best:
        wa, m2 = best["webarena"], best["mind2web"]
        r = stats.compare_unpaired(wa["_trajs"], m2["_trajs"],
                                   label_a="webarena", label_b="mind2web")
        z = r["z_test"]
        verdict = "**significant**" if z["significant_05"] else "not significant"
        lines += [
            "## Sandbox vs realistic — WebArena vs Mind2Web  ·  unpaired",
            f"- WebArena {r['webarena']['rate']} (n={r['webarena']['n']}, "
            f"`{wa['label']}`) vs Mind2Web {r['mind2web']['rate']} "
            f"(n={r['mind2web']['n']}, `{m2['label']}`)",
            f"- difference {r['diff']}  ·  95% CI {r['diff_ci95']}",
            f"- two-proportion z = {z['z']}, **p {fmt_p(z['p_value'])}**  ·  "
            f"Fisher exact p {fmt_p(r['fisher_exact_p'])} → {verdict}",
            "",
        ]
        wrote = True

    if not wrote:
        lines.append("(no comparable run pairs found — need an ON/OFF pair or "
                     "both a WebArena and a Mind2Web run)")
    (out_dir / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    write_significance(df, out_dir)

    print(f"Wrote charts + tables to {out_dir} ({len(df)} run(s)).")
    for f in sorted(os.listdir(out_dir)):
        print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
