"""
run_final_paired_tests.py — Compute paired statistical tests for the final
production wavelet-vs-baseline comparisons and save to models/paired_tests.json.

Pulls per-seed Sharpes directly from training logs and runs:
  • Paired one-sided t-test (treatment > baseline)
  • Wilcoxon signed-rank
  • Cohen's d effect size

Usage:
    python src/utils/run_final_paired_tests.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.utils.statistical_tests import paired_test, sharpes_from_log


LOG_DIR = os.path.join(ROOT, "logs")
OUT_PATH = os.path.join(ROOT, "models", "paired_tests.json")


# (label, baseline log, treatment log)
CASES = [
    ("EURUSD DQN: raw vs wavelet",
     "round1_training/dqn_EURUSD_round1.log",
     "wavelet_full/dqn_EURUSD_wavelet.log"),
    ("EURUSD PPO: raw vs wavelet",
     "round1_training/ppo_EURUSD_round1.log",
     "wavelet_full/ppo_EURUSD_wavelet.log"),
    ("GBPUSD DQN: raw vs wavelet",
     "round1_training/dqn_GBPUSD_round1.log",
     "wavelet_full/dqn_GBPUSD_wavelet.log"),
    ("GBPUSD PPO: raw vs wavelet",
     "round1_training/ppo_GBPUSD_round1.log",
     "wavelet_full/ppo_GBPUSD_wavelet.log"),
    ("AUDUSD DQN: raw vs wavelet",
     "round1_training/dqn_AUDUSD_round1.log",
     "wavelet_training/dqn_AUDUSD_wavelet.log"),
    ("AUDUSD PPO: raw vs wavelet",
     "round1_training/ppo_AUDUSD_round1.log",
     "wavelet_training/ppo_AUDUSD_wavelet.log"),
]


def main():
    results = []
    for label, base_log, treat_log in CASES:
        base_path = os.path.join(LOG_DIR, base_log)
        treat_path = os.path.join(LOG_DIR, treat_log)
        if not (os.path.exists(base_path) and os.path.exists(treat_path)):
            print(f"  SKIP {label}: log missing")
            continue
        base = sharpes_from_log(base_path)[:5]
        treat = sharpes_from_log(treat_path)[:5]
        if len(base) < 2 or len(treat) < 2:
            print(f"  SKIP {label}: not enough seeds (base={len(base)} treat={len(treat)})")
            continue
        n = min(len(base), len(treat))
        r = paired_test(base[:n], treat[:n], label=label)
        results.append({
            "label":        r.label,
            "n":            r.n,
            "baseline":     base[:n],
            "treatment":    treat[:n],
            "mean_lift":    r.mean_lift,
            "sd_lift":      r.sd_lift,
            "t_statistic":  r.t_statistic,
            "t_p":          r.t_p_one_sided,
            "wilcoxon_W":   r.wilcoxon_W,
            "wilcoxon_p":   r.wilcoxon_p_one_sided,
            "cohens_d":     r.cohens_d,
            "effect_size":  r.effect_size,
        })
        print(f"  {label}: lift={r.mean_lift:+.3f}  t-p={r.t_p_one_sided:.4f}  "
              f"d={r.cohens_d:+.2f} ({r.effect_size})")

    out = {
        "tests":        results,
        "method":       "paired one-sided t-test + Wilcoxon signed-rank + Cohen's d",
        "alternative":  "treatment > baseline",
        "matched_by":   "seed index",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
