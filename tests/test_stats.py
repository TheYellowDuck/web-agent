"""Tests for eval.stats — significance tests, effect-size CIs, pass@k.

Values are checked against textbook/closed-form references so the tests pin the
math, not just the plumbing.
"""

from __future__ import annotations

import math

from eval import stats


def _traj(task_id: str, success):
    return {"task_id": task_id, "success": success}


# --- pass@k (unbiased estimator) -------------------------------------------


def test_pass_at_k_basic():
    # 1 success in 3 attempts: pass@1 = 1/3, pass@3 = 1.0.
    assert math.isclose(stats.pass_at_k(3, 1, 1), 1 / 3)
    assert stats.pass_at_k(3, 1, 3) == 1.0
    # pass@2 with 1 success in 3 = 1 - C(2,2)/C(3,2) = 1 - 1/3 = 2/3.
    assert math.isclose(stats.pass_at_k(3, 1, 2), 2 / 3)


def test_pass_at_k_edges():
    assert stats.pass_at_k(3, 0, 2) == 0.0     # no successes ever
    assert stats.pass_at_k(3, 3, 1) == 1.0     # always succeeds
    assert stats.pass_at_k(2, 1, 5) == 1.0     # k clamped to n, and a success exists


# --- Wilson interval --------------------------------------------------------


def test_wilson_interval_known_value():
    lo, hi = stats.wilson_interval(8, 10)  # p=0.8, n=10
    assert 0.49 < lo < 0.51 and 0.94 < hi < 0.95   # ~[0.49, 0.94]
    assert stats.wilson_interval(0, 0) == (0.0, 0.0)


# --- two-proportion z-test --------------------------------------------------


def test_two_proportion_z_clear_difference():
    # 2/24 vs 45/45 — should be wildly significant.
    r = stats.two_proportion_z_test(2, 24, 45, 45)
    assert r["p_value"] < 0.001
    assert r["significant_05"] is True


def test_two_proportion_z_no_difference():
    r = stats.two_proportion_z_test(5, 10, 5, 10)
    assert r["z"] == 0.0
    assert r["p_value"] == 1.0
    assert r["significant_05"] is False


# --- Fisher's exact ---------------------------------------------------------


def test_fisher_exact_symmetry_and_range():
    # A textbook tea-tasting style table; just assert it's a valid p in (0,1].
    p = stats.fisher_exact_two_sided(8, 2, 1, 9)
    assert 0.0 < p <= 1.0
    assert p < 0.05            # strong association
    # Independent-looking table → not significant.
    assert stats.fisher_exact_two_sided(5, 5, 5, 5) > 0.05


# --- Newcombe difference CI -------------------------------------------------


def test_diff_ci_contains_point_and_sign():
    lo, hi = stats.diff_proportion_ci(45, 45, 2, 24)  # big positive diff
    assert lo > 0.0                  # CI excludes 0 → significant gap
    assert lo <= (45 / 45 - 2 / 24) <= hi


def test_diff_ci_overlaps_zero_when_equal():
    lo, hi = stats.diff_proportion_ci(5, 10, 5, 10)
    assert lo < 0.0 < hi


# --- McNemar ----------------------------------------------------------------


def test_mcnemar_no_discordant():
    r = stats.mcnemar_test(0, 0)
    assert r["p_value"] == 1.0
    assert r["significant_05"] is False


def test_mcnemar_exact_small():
    # 0 vs 6 discordant: exact binomial two-sided = 2 * 0.5^6 = 0.03125.
    r = stats.mcnemar_test(0, 6)
    assert r["method"] == "exact binomial"
    assert math.isclose(r["p_value"], 0.03125, rel_tol=1e-6)
    assert r["significant_05"] is True


def test_mcnemar_balanced_not_significant():
    r = stats.mcnemar_test(3, 3)
    assert r["p_value"] == 1.0
    assert r["significant_05"] is False


def test_mcnemar_large_uses_chisquare():
    r = stats.mcnemar_test(30, 10)   # discordant = 40 ≥ 25
    assert r["method"] == "continuity-corrected chi-square"
    assert r["significant_05"] is True


# --- trajectory-level comparisons ------------------------------------------


def test_compare_unpaired_detects_gap():
    a = [_traj(f"wa-{i}", i == 0) for i in range(24)]      # 1/24
    b = [_traj(f"m2w-{i}", True) for i in range(45)]       # 45/45
    r = stats.compare_unpaired(a, b, label_a="webarena", label_b="mind2web")
    assert r["design"] == "unpaired"
    assert r["webarena"]["n"] == 24 and r["mind2web"]["n"] == 45
    assert r["z_test"]["significant_05"] is True
    assert r["diff_ci95"][1] < 0    # webarena - mind2web is solidly negative


def test_compare_paired_mcnemar():
    # Same 6 tasks. ON solves all 6, OFF solves 4 → discordant b=2, c=0.
    on = [_traj(f"t{i}", True) for i in range(6)]
    off = [_traj(f"t{i}", i < 4) for i in range(6)]
    r = stats.compare_paired(on, off, label_a="on", label_b="off")
    assert r["design"] == "paired"
    assert r["n_shared_tasks"] == 6
    assert r["table"]["on_only"] == 2 and r["table"]["off_only"] == 0
    assert r["mcnemar"]["b"] == 2 and r["mcnemar"]["c"] == 0


def test_compare_paired_ignores_unscored():
    on = [_traj("t0", True), _traj("t1", None)]
    off = [_traj("t0", False), _traj("t1", True)]
    r = stats.compare_paired(on, off)
    assert r["n_shared_tasks"] == 1   # t1 dropped (unscored in run A)


def test_per_task_majority_vote():
    # Multi-seed: 2/3 success → counts as a task-level success.
    trajs = [_traj("t0#r1", True), _traj("t0#r2", True), _traj("t0#r3", False)]
    assert stats._per_task_success(trajs) == {"t0": True}
