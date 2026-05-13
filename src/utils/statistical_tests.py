"""
statistical_tests.py — Paired significance tests for capstone model variants.

Ported from the Modern Topics in DS DSP-RL project (May 2026). Used to test
whether a treatment model (e.g. wavelet-denoised AUDUSD) significantly beats
its baseline (e.g. raw-OHLC AUDUSD) by training the same architecture with
multiple random seeds and running paired statistical tests on the Sharpe
ratios.

Tests applied per (variant_pair) tuple:
  • Paired one-sided t-test  (treatment > baseline)
  • Wilcoxon signed-rank     (non-parametric backup)
  • Cohen's d effect size    (small / medium / large)

Usage (programmatic):
    from src.utils.statistical_tests import paired_test
    paired_test(baseline_sharpes=[0.5, 0.6, 0.55],
                treatment_sharpes=[0.9, 1.1, 1.0],
                label="wavelet vs raw")

Usage (CLI):
    python src/utils/statistical_tests.py
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Sequence

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Effect-size helpers
# ─────────────────────────────────────────────────────────────────────────────
def cohens_d_paired(treatment: Sequence[float], control: Sequence[float]) -> float:
    """Cohen's d for paired (matched) samples.

    d = mean(diff) / sd(diff). Magnitude conventions:
      |d| < 0.2 = negligible, < 0.5 = small, < 0.8 = medium, else large.
    """
    diff = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def interpret_d(d: float) -> str:
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


@dataclass
class PairedTestResult:
    label: str
    n: int
    mean_lift: float
    sd_lift: float
    t_statistic: float
    t_p_one_sided: float
    wilcoxon_W: float
    wilcoxon_p_one_sided: float
    cohens_d: float
    effect_size: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{self.label}  (n={self.n})\n"
            f"  Mean Sharpe lift     : {self.mean_lift:+.3f}\n"
            f"  SD of paired diff    : {self.sd_lift:.3f}\n"
            f"  t-statistic          : {self.t_statistic:+.3f}\n"
            f"  One-sided p (t-test) : {self.t_p_one_sided:.4f}\n"
            f"  Wilcoxon W           : {self.wilcoxon_W}\n"
            f"  One-sided p (Wilcoxon): {self.wilcoxon_p_one_sided}\n"
            f"  Cohen's d (paired)   : {self.cohens_d:+.3f}  ({self.effect_size} effect)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core paired test
# ─────────────────────────────────────────────────────────────────────────────
def paired_test(
    baseline_sharpes: Sequence[float],
    treatment_sharpes: Sequence[float],
    label: str = "treatment vs baseline",
) -> PairedTestResult:
    """Run all paired tests in one call.

    Both inputs must be matched: index i in baseline corresponds to index i in
    treatment (same seed, same pair, same period — whatever the matching is).

    Returns a `PairedTestResult` with t-stat, Wilcoxon, Cohen's d, and effect
    size label. Prints a friendly summary too.
    """
    base = np.asarray(baseline_sharpes, dtype=float)
    treat = np.asarray(treatment_sharpes, dtype=float)
    if len(base) != len(treat):
        raise ValueError(
            f"paired_test requires equal-length arrays "
            f"(got {len(base)} vs {len(treat)})"
        )
    if len(base) < 2:
        raise ValueError(f"paired_test requires n >= 2 (got n={len(base)})")

    # Paired t-test (one-sided: treatment > baseline)
    t_stat, p_two = stats.ttest_rel(treat, base)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    # Wilcoxon signed-rank
    try:
        w_stat, w_p = stats.wilcoxon(treat, base, alternative="greater")
        w_stat = float(w_stat)
        w_p = float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    diff = treat - base
    d = cohens_d_paired(treat, base)

    return PairedTestResult(
        label=label,
        n=len(base),
        mean_lift=float(np.mean(diff)),
        sd_lift=float(np.std(diff, ddof=1)),
        t_statistic=float(t_stat),
        t_p_one_sided=float(p_one),
        wilcoxon_W=w_stat,
        wilcoxon_p_one_sided=w_p,
        cohens_d=float(d),
        effect_size=interpret_d(d),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: pull sharpes from training logs of demo.py multi-seed runs
# ─────────────────────────────────────────────────────────────────────────────
import re

_SHARPE_LINE = re.compile(r"Sharpe:\s*([+-]?\d+\.\d+)")
_RETURN_LINE = re.compile(r"Return:\s*([+-]?\d+\.\d+)%")


def sharpes_from_log(log_path: str) -> List[float]:
    """Extract per-seed validation Sharpes from a demo.py --seeds N log file.

    Each seed prints a 'Return: ...  Sharpe: ...' line during multi-seed
    training. Returns a list of Sharpe values in encounter order.
    """
    sharpes = []
    with open(log_path) as f:
        for line in f:
            m = _SHARPE_LINE.search(line)
            if m and "Best seed" not in line:
                # Avoid the 'Best seed: ... (Sharpe: ...)' summary line
                sharpes.append(float(m.group(1)))
    # Drop the final "Sharpe Ratio" line printed in the per-strategy table;
    # the multi-seed loop prints exactly N lines before that table.
    return sharpes


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo: compare two log files
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Paired test: treatment log vs baseline log")
    p.add_argument("--baseline", required=True, help="Path to baseline training log (.log)")
    p.add_argument("--treatment", required=True, help="Path to treatment training log (.log)")
    p.add_argument("--label", default="treatment vs baseline", help="Label for output")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    base_sharpes = sharpes_from_log(args.baseline)
    treat_sharpes = sharpes_from_log(args.treatment)

    if len(base_sharpes) != len(treat_sharpes):
        # Trim to min length so matched-pairs assumption holds
        n = min(len(base_sharpes), len(treat_sharpes))
        print(f"Note: log lengths differ ({len(base_sharpes)} vs {len(treat_sharpes)}). "
              f"Using first {n} of each.")
        base_sharpes = base_sharpes[:n]
        treat_sharpes = treat_sharpes[:n]

    print(f"Baseline Sharpes : {base_sharpes}")
    print(f"Treatment Sharpes: {treat_sharpes}")
    print()
    result = paired_test(base_sharpes, treat_sharpes, label=args.label)
    print(result)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
