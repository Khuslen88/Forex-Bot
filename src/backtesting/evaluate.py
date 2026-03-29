"""
evaluate.py — Compute and compare performance metrics from backtest results.

Usage
-----
from src.backtesting.evaluate import compute_metrics, print_comparison

metrics = compute_metrics(equity_curve, trades)
print_comparison({"DQN": dqn_metrics, "Buy&Hold": bah_metrics, "SMA": sma_metrics})
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    equity_curve: List[float],
    trades: List[Dict],
) -> Dict[str, float]:
    """
    Compute standard trading performance metrics from backtest results.

    Parameters
    ----------
    equity_curve : list of portfolio values, one per day
                   e.g. [10000.0, 10050.3, 9987.1, ...]
    trades       : list of trade dicts, each with a 'pnl' key
                   e.g. [{"pnl": 120.5, "entry": 1.08, "exit": 1.09, ...}, ...]

    Returns
    -------
    dict with keys: total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor
    """
    equity = np.array(equity_curve, dtype=np.float64)

    daily_returns = np.diff(equity) / equity[:-1]

    total_return  = equity[-1] / equity[0] - 1

    sharpe_ratio  = (
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    )

    running_peak = np.maximum.accumulate(equity)
    drawdowns    = (equity - running_peak) / running_peak
    max_drawdown = float(np.min(drawdowns))

    if len(trades) == 0:
        win_rate      = 0.0
        profit_factor = 0.0
    else:
        pnls    = np.array([t["pnl"] for t in trades], dtype=np.float64)
        wins    = pnls[pnls > 0]
        losses  = pnls[pnls < 0]
        win_rate      = len(wins) / len(pnls)
        profit_factor = float(wins.sum() / (abs(losses.sum()) + 1e-8)) if len(losses) > 0 else 0.0

    return {
        "total_return":  total_return,
        "sharpe_ratio":  sharpe_ratio,
        "max_drawdown":  max_drawdown,
        "win_rate":      win_rate,
        "profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------

def print_comparison(results: Dict[str, Dict]) -> None:
    """
    Print a side-by-side comparison table of strategy metrics.

    Parameters
    ----------
    results : dict mapping strategy name → metrics dict from compute_metrics()
              e.g. {"DQN": {...}, "Buy&Hold": {...}, "SMA Crossover": {...}}
    """
    strategies = list(results.keys())
    metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"]
    labels  = ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)", "Profit Factor"]

    col_w  = 18
    header = f"{'Metric':<22}" + "".join(f"{s:>{col_w}}" for s in strategies)
    divider = "=" * len(header)

    print(f"\n{divider}")
    print(header)
    print(divider)

    for metric, label in zip(metrics, labels):
        row = f"{label:<22}"
        for s in strategies:
            val = results[s].get(metric, 0.0)
            if metric in ("total_return", "max_drawdown", "win_rate"):
                row += f"{val * 100:>{col_w}.2f}"
            else:
                row += f"{val:>{col_w}.3f}"
        print(row)

    print(divider + "\n")


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_test(
    df: pd.DataFrame,
    train_fn: Callable,
    run_fn: Callable,
    n_splits: int = 5,
    train_ratio: float = 0.6,
) -> Dict:
    """
    Walk-forward validation for time-series trading strategies.

    Splits the data into n_splits windows. Each window has a training
    segment followed by a test segment. The model is retrained from
    scratch on each training segment and evaluated on the following
    test segment — ensuring no future data leaks into training.

    Parameters
    ----------
    df          : Full DataFrame (train + test combined), sorted by date
    train_fn    : Callable(train_df) → model
    run_fn      : Callable(model, test_df) → (equity_curve, trades)
    n_splits    : Number of walk-forward folds
    train_ratio : Fraction of each fold used for training (rest = test)

    Returns
    -------
    dict with keys:
      folds      : list of per-fold metric dicts
      aggregated : averaged metrics across all folds
      equity_curves : list of equity curves, one per fold
    """
    total_rows  = len(df)
    fold_size   = total_rows // n_splits
    fold_results = []
    all_equity  = []

    print(f"\n  Walk-forward: {n_splits} folds × {fold_size} rows "
          f"(train={int(train_ratio*100)}% / test={int((1-train_ratio)*100)}%)")

    for i in range(n_splits):
        fold_start = i * fold_size
        fold_end   = fold_start + fold_size if i < n_splits - 1 else total_rows
        fold_df    = df.iloc[fold_start:fold_end].copy().reset_index(drop=True)

        split_idx  = int(len(fold_df) * train_ratio)
        train_df   = fold_df.iloc[:split_idx]
        test_df    = fold_df.iloc[split_idx:].reset_index(drop=True)

        if len(train_df) < 50 or len(test_df) < 10:
            print(f"  Fold {i+1}: skipped (too small)")
            continue

        model          = train_fn(train_df)
        equity, trades = run_fn(model, test_df)
        metrics        = compute_metrics(equity, trades)

        fold_results.append(metrics)
        all_equity.append(equity)

        date_start = test_df["Date"].iloc[0].strftime("%Y-%m-%d")
        date_end   = test_df["Date"].iloc[-1].strftime("%Y-%m-%d")
        print(f"  Fold {i+1} test [{date_start} → {date_end}]  "
              f"return={metrics['total_return']*100:+.1f}%  "
              f"sharpe={metrics['sharpe_ratio']:.2f}  "
              f"dd={metrics['max_drawdown']*100:.1f}%")

    if not fold_results:
        return {"folds": [], "aggregated": {}, "equity_curves": []}

    # Average each metric across folds
    keys = fold_results[0].keys()
    aggregated = {k: float(np.mean([f[k] for f in fold_results])) for k in keys}

    print(f"\n  Walk-forward aggregate  "
          f"return={aggregated['total_return']*100:+.1f}%  "
          f"sharpe={aggregated['sharpe_ratio']:.2f}  "
          f"dd={aggregated['max_drawdown']*100:.1f}%  "
          f"win_rate={aggregated['win_rate']*100:.1f}%")

    return {
        "folds":        fold_results,
        "aggregated":   aggregated,
        "equity_curves": all_equity,
    }


def meets_targets(metrics: Dict[str, float]) -> Dict[str, bool]:
    """
    Check whether DQN results meet the project proposal targets.

    Targets (from config/settings.py):
      Sharpe Ratio   > 1.0
      Max Drawdown   < 20%   (note: max_drawdown is stored as a negative fraction)
      Win Rate       > 45%
      Profit Factor  > 1.2
    """
    return {
        "sharpe_ratio ≥ 1.0":    metrics["sharpe_ratio"]  >= 1.0,
        "max_drawdown ≤ -20%":   metrics["max_drawdown"]  >= -0.20,
        "win_rate ≥ 45%":        metrics["win_rate"]      >= 0.45,
        "profit_factor ≥ 1.2":   metrics["profit_factor"] >= 1.2,
    }
