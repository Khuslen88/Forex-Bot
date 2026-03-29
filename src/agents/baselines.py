"""
baselines.py — Simple baseline trading strategies for comparison against DQN.

Three agents:
  - BuyAndHold    : enter long on day 1, hold until the end
  - SMAcrossover  : long when SMA_50 > SMA_200, flat otherwise
  - RandomAgent   : random Buy / Sell / Hold each day (lower-bound baseline)

All return (equity_curve: list[float], trades: list[dict]) so results
can be fed directly into backtesting/evaluate.py compute_metrics().
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Buy-and-Hold
# ---------------------------------------------------------------------------

def run_buy_and_hold(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    trading_cost: float = 0.0001,
) -> Tuple[List[float], List[Dict]]:
    """
    Simplest possible strategy: go long on day 1, never exit.

    Cost is applied twice (entry + final exit) to keep accounting consistent
    with the other strategies.
    """
    prices  = df["Close"].values
    balance = initial_balance * (1.0 - trading_cost)   # entry cost on day 1
    equity  = [initial_balance, balance]

    for i in range(len(prices) - 1):
        daily_return = (prices[i + 1] - prices[i]) / prices[i]
        balance     *= 1.0 + daily_return
        equity.append(balance)

    balance *= (1.0 - trading_cost)                    # exit cost on last day
    equity[-1] = balance

    trades = [{
        "entry":     prices[0],
        "exit":      prices[-1],
        "direction": "long",
        "pnl":       balance - initial_balance,
    }]

    return equity, trades


# ---------------------------------------------------------------------------
# SMA Crossover
# ---------------------------------------------------------------------------

def run_sma_crossover(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    trading_cost: float = 0.0001,
) -> Tuple[List[float], List[Dict]]:
    """
    Classic trend-following strategy:
      - Long  when SMA_50 > SMA_200  (golden cross = uptrend)
      - Flat  when SMA_50 < SMA_200  (death  cross = downtrend)

    Requires df to have 'Close', 'SMA_50', 'SMA_200' columns
    (produced by src/features/indicators.py).
    """
    prices  = df["Close"].values
    sma50   = df["SMA_50"].values
    sma200  = df["SMA_200"].values

    balance  = initial_balance
    position = 0       # 0 = flat, 1 = long
    equity   = [balance]
    trades: List[Dict] = []

    entry_price   = None
    entry_balance = None

    for i in range(len(prices) - 1):
        signal = 1 if sma50[i] > sma200[i] else 0

        # ── Handle position change ────────────────────────────────────────
        if signal != position:
            balance -= abs(balance) * trading_cost          # pay spread cost

            if position == 1 and entry_price is not None:   # close existing long
                trades.append({
                    "entry":     entry_price,
                    "exit":      prices[i],
                    "direction": "long",
                    "pnl":       balance - entry_balance,
                })

            position = signal
            if position == 1:                               # opening new long
                entry_price   = prices[i]
                entry_balance = balance

        # ── Mark-to-market for held long ─────────────────────────────────
        if position == 1:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            balance     *= 1.0 + daily_return

        equity.append(balance)

    # Close any open position at the end
    if position == 1 and entry_price is not None:
        balance -= balance * trading_cost
        trades.append({
            "entry":     entry_price,
            "exit":      prices[-1],
            "direction": "long",
            "pnl":       balance - entry_balance,
        })
        equity[-1] = balance

    return equity, trades


# ---------------------------------------------------------------------------
# Random Agent
# ---------------------------------------------------------------------------

def run_random_agent(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    trading_cost: float = 0.0001,
    seed: int = 42,
) -> Tuple[List[float], List[Dict]]:
    """
    Makes random Buy / Sell / Hold decisions each day.

    Used as a lower-bound sanity check — the DQN should clearly beat this.
    Fixed seed ensures reproducibility across demo runs.
    """
    prices   = df["Close"].values
    rng      = np.random.default_rng(seed)
    balance  = initial_balance
    position = 0   # -1 = short, 0 = flat, 1 = long
    equity   = [balance]
    trades: List[Dict] = []

    for i in range(len(prices) - 1):
        action = int(rng.integers(0, 3))   # 0=flat, 1=long, 2=short
        target = {0: 0, 1: 1, 2: -1}[action]

        if target != position:
            balance  -= abs(balance) * trading_cost
            position  = target

        daily_return = (prices[i + 1] - prices[i]) / prices[i]
        pnl_frac     = position * daily_return
        balance     *= 1.0 + pnl_frac

        equity.append(balance)
        if pnl_frac != 0:
            trades.append({"pnl": pnl_frac * balance})

    return equity, trades
