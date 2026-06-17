"""Significance tests + effect-size intervals for honest ablation comparisons.

Pure stdlib (``math`` only) — no SciPy dependency, so it runs anywhere the agent
runs and stays installable as a single ``pip install -e .``.

Why this exists: the project's claims are *comparative* ("reflection ON ≈ OFF",
"the sandbox-vs-realistic gap is real"). The tempting shortcut — "do the two 95%
confidence intervals overlap?" — is a poor test: non-overlapping CIs imply
significance, but *overlapping* CIs very often hide a significant difference, so
eyeballing overlap systematically under-detects real effects. This module backs
those claims with the correct test for each design:

  * unpaired (different task sets, e.g. WebArena vs Mind2Web):
        two-proportion z-test + Fisher's exact + Newcombe difference-CI
  * paired   (same task set, e.g. reflect ON vs OFF on the matched set):
        McNemar's test (exact binomial for small discordant counts)

It also implements the **unbiased pass@k** estimator (Kulal et al. 2019, as used
by HumanEval) — the rigorous version of "solved at least once".

Everything operates on the trajectory dicts written to JSONL, so comparisons are
computed entirely offline from saved runs (no re-running, no API cost).
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Low-level building blocks (kept here so metrics.py imports from one place)
# ---------------------------------------------------------------------------


def normal_cdf(z: float) -> float:
    """Standard-normal CDF via the error function (exact, stdlib)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion — honest error bars for small N."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Kulal et al. 2019): probability that at least one of k
    samples drawn (without replacement) from ``n`` attempts with ``c`` successes
    passes. ``1 - C(n-c, k) / C(n, k)``."""
    if k <= 0 or n <= 0:
        return 0.0
    if k > n:
        k = n
    if n - c < k:  # not enough failures to fill a k-sample → guaranteed a hit
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# Two-proportion (unpaired) tests
# ---------------------------------------------------------------------------


def two_proportion_z_test(s1: int, n1: int, s2: int, n2: int) -> dict[str, Any]:
    """Pooled two-sided z-test for H0: p1 == p2 (independent samples)."""
    if n1 == 0 or n2 == 0:
        return {"test": "two_proportion_z", "p_value": None, "z": None,
                "note": "empty sample"}
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        # Both groups are all-success or all-failure with identical rates.
        z = 0.0 if p1 == p2 else math.inf
    else:
        z = (p1 - p2) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z))) if math.isfinite(z) else 0.0
    return {
        "test": "two_proportion_z",
        "p1": round(p1, 4), "p2": round(p2, 4),
        "z": round(z, 4) if math.isfinite(z) else None,
        "p_value": round(p_value, 5),
        "significant_05": p_value < 0.05,
    }


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p-value for the 2x2 table [[a, b], [c, d]].

    Sums the hypergeometric probability of every table (with the same margins)
    no more likely than the observed one. Exact — preferred over the z-test when
    counts are small.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c
    row2 = c + d

    def hyper(x: int) -> float:
        # P(top-left = x) given fixed margins.
        lo = max(0, col1 - row2)
        hi = min(row1, col1)
        if x < lo or x > hi:
            return 0.0
        return (
            math.comb(row1, x)
            * math.comb(row2, col1 - x)
            / math.comb(n, col1)
        )

    p_obs = hyper(a)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    total = 0.0
    for x in range(lo, hi + 1):
        px = hyper(x)
        if px <= p_obs * (1 + 1e-9):
            total += px
    return round(min(1.0, total), 5)


def diff_proportion_ci(
    s1: int, n1: int, s2: int, n2: int, z: float = 1.96
) -> tuple[float, float]:
    """Newcombe (1998) score CI for the difference p1 - p2 (independent samples).

    Combines each proportion's Wilson interval by square-and-add — well-behaved
    near 0/1 and for small N, where the naive Wald CI for a difference fails.
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 0.0)
    p1, p2 = s1 / n1, s2 / n2
    l1, u1 = wilson_interval(s1, n1, z)
    l2, u2 = wilson_interval(s2, n2, z)
    lower = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (round(max(-1.0, lower), 4), round(min(1.0, upper), 4))


# ---------------------------------------------------------------------------
# McNemar (paired) test
# ---------------------------------------------------------------------------


def mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """McNemar's test on the discordant pairs of two paired binary classifiers.

    ``b`` = count where A succeeded and B failed; ``c`` = A failed, B succeeded.
    Concordant pairs (both right / both wrong) carry no information about the
    *difference* and are excluded by design. Uses the exact binomial test when
    the discordant total is small (<25), else the continuity-corrected
    chi-square — the standard guidance.
    """
    nd = b + c
    if nd == 0:
        return {"test": "mcnemar", "b": b, "c": c, "p_value": 1.0,
                "method": "no discordant pairs", "significant_05": False}
    if nd < 25:
        # Exact two-sided binomial against p=0.5.
        k = min(b, c)
        tail = sum(math.comb(nd, i) for i in range(0, k + 1)) * (0.5 ** nd)
        p_value = min(1.0, 2.0 * tail)
        method = "exact binomial"
    else:
        stat = (abs(b - c) - 1) ** 2 / nd  # continuity-corrected, 1 df
        p_value = math.erfc(math.sqrt(stat / 2.0))  # chi-square(1) survival
        method = "continuity-corrected chi-square"
    return {
        "test": "mcnemar", "b": b, "c": c, "discordant": nd,
        "p_value": round(p_value, 5), "method": method,
        "significant_05": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Trajectory-level comparisons (operate on JSONL dicts)
# ---------------------------------------------------------------------------


def _base_task_id(task_id: str) -> str:
    return re.sub(r"#r\d+$", "", task_id or "")


def _per_task_success(trajs: list[dict[str, Any]]) -> dict[str, bool]:
    """Collapse a run to one binary outcome per base task (majority of attempts,
    ties → success), ignoring unscored trajectories."""
    agg: dict[str, list[int]] = {}
    for t in trajs:
        if t.get("success") is None:
            continue
        agg.setdefault(_base_task_id(t["task_id"]), []).append(1 if t.get("success") else 0)
    return {tid: (sum(v) / len(v)) >= 0.5 for tid, v in agg.items() if v}


def _counts(trajs: list[dict[str, Any]]) -> tuple[int, int]:
    """(successes, n) over scored trajectories."""
    scored = [t for t in trajs if t.get("success") is not None]
    return sum(1 for t in scored if t.get("success")), len(scored)


def compare_unpaired(
    a: list[dict[str, Any]], b: list[dict[str, Any]],
    *, label_a: str = "A", label_b: str = "B",
) -> dict[str, Any]:
    """Compare two runs on *different* task sets (e.g. WebArena vs Mind2Web)."""
    s1, n1 = _counts(a)
    s2, n2 = _counts(b)
    return {
        "design": "unpaired",
        label_a: {"successes": s1, "n": n1, "rate": round(s1 / n1, 4) if n1 else None,
                  "ci95": list(wilson_interval(s1, n1))},
        label_b: {"successes": s2, "n": n2, "rate": round(s2 / n2, 4) if n2 else None,
                  "ci95": list(wilson_interval(s2, n2))},
        "diff": round((s1 / n1) - (s2 / n2), 4) if n1 and n2 else None,
        "diff_ci95": list(diff_proportion_ci(s1, n1, s2, n2)),
        "z_test": two_proportion_z_test(s1, n1, s2, n2),
        "fisher_exact_p": fisher_exact_two_sided(s1, n1 - s1, s2, n2 - s2),
    }


def compare_paired(
    a: list[dict[str, Any]], b: list[dict[str, Any]],
    *, label_a: str = "A", label_b: str = "B",
) -> dict[str, Any]:
    """Compare two runs on the *same* tasks (e.g. reflect ON vs OFF) — McNemar."""
    sa = _per_task_success(a)
    sb = _per_task_success(b)
    shared = sorted(set(sa) & set(sb))
    n00 = n01 = n10 = n11 = 0
    for tid in shared:
        ra, rb = sa[tid], sb[tid]
        if ra and rb:
            n11 += 1
        elif ra and not rb:
            n10 += 1
        elif not ra and rb:
            n01 += 1
        else:
            n00 += 1
    rate_a = sum(sa[t] for t in shared) / len(shared) if shared else None
    rate_b = sum(sb[t] for t in shared) / len(shared) if shared else None
    return {
        "design": "paired",
        "n_shared_tasks": len(shared),
        f"{label_a}_rate": round(rate_a, 4) if rate_a is not None else None,
        f"{label_b}_rate": round(rate_b, 4) if rate_b is not None else None,
        "table": {"both_pass": n11, f"{label_a}_only": n10,
                  f"{label_b}_only": n01, "both_fail": n00},
        "mcnemar": mcnemar_test(n10, n01),
    }


# ---------------------------------------------------------------------------
# CLI: compare two JSONL runs offline
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from eval.metrics import load_trajectories

    p = argparse.ArgumentParser(
        description="Significance test between two trajectory runs (offline)."
    )
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--paired", action="store_true",
                   help="same task set (e.g. reflect ON vs OFF) → McNemar; "
                        "default is unpaired (two-proportion / Fisher).")
    args = p.parse_args(argv)

    a = load_trajectories(Path(args.run_a))
    b = load_trajectories(Path(args.run_b))
    la, lb = Path(args.run_a).stem, Path(args.run_b).stem
    fn = compare_paired if args.paired else compare_unpaired
    print(json.dumps(fn(a, b, label_a=la, label_b=lb), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
