"""
ensemble.py — Multi-timeframe model ensemble (confirmation strategy).

For each pair, run a daily-paced backtest where the action is taken only when
the daily AND hourly models agree. If they disagree, the bot goes FLAT.

The hypothesis: requiring agreement filters out low-conviction signals,
boosting win rate at the cost of fewer trades.

Usage (programmatic):
    from src.ensemble import run_ensemble_for_pair
    metrics = run_ensemble_for_pair("EURUSD", version="v2")

Usage (CLI):
    python src/ensemble.py --pair EURUSD --version v2
    python src/ensemble.py --pair all --version v2
"""

import os
import sys
import json
import argparse
from functools import partial

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import DQN, PPO

from src.features.indicators       import load_pair
from src.environment.feature_env   import ForexFeatureEnv
from src.backtesting.evaluate      import compute_metrics
from config.settings               import (TRAIN_TEST_SPLIT, TIMEFRAME_CONFIGS, MODELS_PATH,
                                            ENV_V2_DEFAULTS, PIP_SIZES, SPREAD_PIPS,
                                            FOREX_PAIRS)

DATA_DIR  = os.path.join(ROOT, "data", "raw")
ECON_PATH = os.path.join(ROOT, "data", "economic", "fred_data.csv")
REGISTRY  = os.path.join(MODELS_PATH, "registry.json")


# ── Loading helpers ──────────────────────────────────────────────────────────

def _load_test_df(pair: str, timeframe: str) -> pd.DataFrame:
    cfg = TIMEFRAME_CONFIGS[timeframe]
    raw = pd.read_csv(os.path.join(DATA_DIR, cfg["data_file"]))
    df = load_pair(raw, pair, econ_path=ECON_PATH if os.path.exists(ECON_PATH) else None)
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def _load_model(pair: str, timeframe: str, agent: str, version: str = "v1"):
    """Try to load the model for a (pair, timeframe, agent, version). Returns None if missing."""
    tf_suffix  = "" if timeframe == "1d" else f"_{timeframe}"
    ver_suffix = "" if version   == "v1" else f"_{version}"
    fname = f"{agent.lower()}_feature_{pair}{tf_suffix}{ver_suffix}.zip"
    path = os.path.join(MODELS_PATH, fname)
    if not os.path.exists(path):
        return None, fname
    loader = PPO if agent.upper() == "PPO" else DQN
    return loader.load(path), fname


def _make_env_factory(timeframe: str, version: str, pair: str):
    """Build the same env factory demo.py uses, so SL/TP/cost match training."""
    cfg = TIMEFRAME_CONFIGS[timeframe]
    kwargs = dict(
        stop_loss=cfg["stop_loss_pct"],
        take_profit=cfg["take_profit_pct"],
        min_hold_steps=cfg["min_hold_steps"],
    )
    if version == "v2":
        kwargs.update(
            atr_mult_sl=ENV_V2_DEFAULTS["atr_mult_sl"],
            atr_mult_tp=ENV_V2_DEFAULTS["atr_mult_tp"],
            spread_pips=SPREAD_PIPS.get(pair, 1.0),
            slippage_pips=ENV_V2_DEFAULTS["slippage_pips"],
            pip_size=PIP_SIZES.get(pair, 0.0001),
        )
    return partial(ForexFeatureEnv, **kwargs)


def _best_pair_meta(pair: str, timeframe: str, version: str):
    """Return (agent, sharpe) of the best v{version} model for (pair, timeframe).

    Falls back to v1 if v2 models aren't trained yet for that combo.
    """
    if not os.path.exists(REGISTRY):
        return None, None
    with open(REGISTRY) as f:
        reg = json.load(f)
    cands = [m for m in reg["models"]
             if m["pair"] == pair
             and m["timeframe"] == timeframe
             and m.get("version", "v1") == version]
    if not cands:
        # fall back to v1 if requested version missing
        cands = [m for m in reg["models"]
                 if m["pair"] == pair and m["timeframe"] == timeframe]
        if not cands:
            return None, None
    best = max(cands, key=lambda m: m["sharpe_ratio"])
    return best["agent"], best["sharpe_ratio"]


# ── Aligning hourly observations to daily bars ──────────────────────────────

def _hourly_index_for_each_daily(daily_df: pd.DataFrame, hourly_df: pd.DataFrame) -> np.ndarray:
    """For each daily bar at date D, find the index of the most recent hourly bar
    on or before D (so the hourly model sees no future info)."""
    # Both have a Date column; convert to comparable form (drop tz on hourly)
    h_dates = pd.to_datetime(hourly_df["Date"])
    if h_dates.dt.tz is not None:
        h_dates = h_dates.dt.tz_localize(None)
    h_arr = h_dates.values

    d_arr = pd.to_datetime(daily_df["Date"]).values

    # searchsorted gives insertion point — subtract 1 to get last <= daily_date
    idx = np.searchsorted(h_arr, d_arr, side="right") - 1
    return idx


def _build_obs(env: ForexFeatureEnv, df: pd.DataFrame, idx: int) -> np.ndarray:
    """Mimic ForexFeatureEnv._get_obs() at an arbitrary index without stepping."""
    window = env.window
    if idx < window or idx >= len(df):
        return None

    prices = df["Close"].values.astype(np.float64)
    window_prices = prices[idx - window: idx + 1]
    returns = np.diff(window_prices) / window_prices[:-1]
    rolling_vol = float(np.std(returns) * np.sqrt(252))
    base = np.append(returns, [rolling_vol, 0.0]).astype(np.float32)

    close = float(prices[idx]) + 1e-8
    extra = [
        float(df["RSI"].iloc[idx]) / 100.0,
        float(df["MACD"].iloc[idx]) / close,
        float(df["MACD_signal"].iloc[idx]) / close,
        float(df["BB_width"].iloc[idx]),
        float(df["ATR"].iloc[idx]) / close,
        float(df["ADX"].iloc[idx]) / 100.0,
        float(df["SMA_ratio"].iloc[idx]) - 1.0,
    ]
    if "rate_diff" in df.columns:
        extra += [float(df["rate_diff"].iloc[idx]),
                  float(df["yield_curve"].iloc[idx]),
                  float(df["rate_diff_ma"].iloc[idx])]
    if "sentiment_score" in df.columns:
        extra += [float(df["sentiment_score"].iloc[idx]),
                  float(df["sentiment_vol"].iloc[idx]),
                  float(df["news_volume"].iloc[idx])]

    return np.concatenate([base, np.array(extra, dtype=np.float32)])


# ── Core ensemble runner ─────────────────────────────────────────────────────

def run_ensemble_for_pair(pair: str, version: str = "v2") -> dict | None:
    """Run a daily-paced ensemble backtest for one pair.

    Strategy: at each daily bar, the daily model AND the hourly model must
    agree on direction. If not, go FLAT.
    """
    # Pick best agent per timeframe (e.g. DQN daily + DQN 1H, or DQN+PPO mixed)
    daily_agent, _  = _best_pair_meta(pair, "1d", version)
    hourly_agent, _ = _best_pair_meta(pair, "1h", version)

    if daily_agent is None or hourly_agent is None:
        print(f"  {pair}: missing model (daily={daily_agent}, hourly={hourly_agent})")
        return None

    daily_model, dname = _load_model(pair, "1d", daily_agent, version)
    hourly_model, hname = _load_model(pair, "1h", hourly_agent, version)
    if daily_model is None or hourly_model is None:
        print(f"  {pair}: model files missing ({dname}, {hname})")
        return None

    daily_df  = _load_test_df(pair, "1d")
    hourly_df = _load_test_df(pair, "1h")

    # Build env on daily data using v2 settings (so cost/SL match)
    env_factory = _make_env_factory("1d", version, pair)
    env = env_factory(daily_df)
    obs, _ = env.reset()

    # Pre-compute hourly index for each daily step
    h_idx_for_d = _hourly_index_for_each_daily(daily_df, hourly_df)

    # Step through, querying both models at each bar
    trades, equity = [], [env.initial_balance]
    prev_pos = 0
    entry_price = None
    entry_balance = None

    while True:
        # Daily action from the env's current observation
        daily_action, _ = daily_model.predict(obs, deterministic=True)
        daily_action = int(daily_action)

        # Hourly action — build obs at the matching hourly bar
        d_step = env.current_step
        h_step = int(h_idx_for_d[d_step]) if d_step < len(h_idx_for_d) else -1
        h_obs = _build_obs(env, hourly_df, h_step) if h_step > 0 else None

        if h_obs is None:
            ensemble_action = daily_action  # not enough hourly history — default to daily
        else:
            # Resize hourly obs to match hourly model's expected dim
            expected = int(hourly_model.observation_space.shape[0])
            if len(h_obs) < expected:
                h_obs = np.concatenate([h_obs, np.zeros(expected - len(h_obs), dtype=np.float32)])
            elif len(h_obs) > expected:
                h_obs = h_obs[:expected]
            hourly_action, _ = hourly_model.predict(h_obs, deterministic=True)
            hourly_action = int(hourly_action)

            # Confirmation rule: trade only if both agree, else FLAT
            ensemble_action = daily_action if daily_action == hourly_action else 0

        obs, _r, terminated, truncated, info = env.step(ensemble_action)
        equity.append(info["balance"])
        new_pos = info["position"]

        if new_pos != prev_pos:
            if prev_pos != 0 and entry_price is not None:
                trades.append({
                    "entry":     entry_price,
                    "exit":      daily_df["Close"].iloc[min(env.current_step - 1, len(daily_df) - 1)],
                    "direction": "long" if prev_pos == 1 else "short",
                    "pnl":       info["balance"] - entry_balance,
                })
            if new_pos != 0:
                entry_price = daily_df["Close"].iloc[min(env.current_step - 1, len(daily_df) - 1)]
                entry_balance = info["balance"]

        prev_pos = new_pos
        if terminated or truncated:
            break

    if prev_pos != 0 and entry_price is not None:
        trades.append({
            "entry":     entry_price,
            "exit":      daily_df["Close"].iloc[-1],
            "direction": "long" if prev_pos == 1 else "short",
            "pnl":       equity[-1] - entry_balance,
        })

    metrics = compute_metrics(equity, trades)
    metrics.update({
        "pair":         pair,
        "version":      version,
        "daily_agent":  daily_agent,
        "hourly_agent": hourly_agent,
        "n_trades":     len(trades),
        "final_balance": float(equity[-1]),
    })
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Ensemble (daily+1H confirmation) backtest")
    p.add_argument("--pair",    default="all",
                   help="Currency pair, or 'all' for every pair (default: all)")
    p.add_argument("--version", default="v1", choices=["v1", "v2"],
                   help="Env version of underlying models (default: v1)")
    args = p.parse_args()

    pairs = FOREX_PAIRS if args.pair.lower() == "all" else [args.pair.upper()]

    print(f"Ensemble backtest — version: {args.version}")
    print(f"{'Pair':<8} {'Daily':<6} {'Hourly':<7} {'Return':>8}  {'Sharpe':>7}  "
          f"{'WinRate':>7}  {'PF':>5}  {'DD':>7}  {'Trades':>6}")
    print("-" * 80)

    rows = []
    for pair in pairs:
        m = run_ensemble_for_pair(pair, version=args.version)
        if m is None:
            continue
        rows.append(m)
        print(f"{m['pair']:<8} {m['daily_agent']:<6} {m['hourly_agent']:<7} "
              f"{m['total_return']*100:>+7.2f}%  {m['sharpe_ratio']:>+7.3f}  "
              f"{m['win_rate']*100:>6.1f}%  {m['profit_factor']:>5.2f}  "
              f"{m['max_drawdown']*100:>+6.1f}%  {m['n_trades']:>6d}")

    # Persist to registry sidecar
    out_path = os.path.join(MODELS_PATH, f"ensemble_{args.version}.json")
    with open(out_path, "w") as f:
        json.dump({"results": rows, "version": args.version}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
