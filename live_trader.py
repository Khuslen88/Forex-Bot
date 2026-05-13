"""
live_trader.py — Connect the trained DQN model to a demo trading account.

Supports two broker backends:
  --broker paper  (default — paper trading with Yahoo Finance prices, works on Mac)
  --broker mt5    (MetaTrader 5 — requires Windows + MT5 terminal running)

Usage:
  # Paper trading (default) — dry run:
  python live_trader.py --pair EURUSD --dry-run

  # Paper trading — execute trade:
  python live_trader.py --pair EURUSD

  # Check paper account status:
  python live_trader.py --pair EURUSD --status

  # Reset paper account:
  python live_trader.py --reset

  # Run continuously, checking every 24 hours (default):
  python live_trader.py --pair all --loop

  # Loop with custom interval (e.g. every 1 hour):
  python live_trader.py --pair all --loop --interval 3600

  # Use a specific model:
  python live_trader.py --pair EURUSD --model models/dqn_feature_EURUSD.zip
"""

import os
import sys
import time
import json
import signal
import argparse
import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import DQN, PPO
from src.features.indicators import add_indicators, load_econ_features
from src.features.sentiment import add_sentiment_to_df, HAS_TEXTBLOB
from config.settings import LIVE_TRADE_UNITS, MODELS_PATH, FOREX_PAIRS

ROOT = os.path.dirname(os.path.abspath(__file__))
ECON_PATH = os.path.join(ROOT, "data", "economic", "fred_data.csv")
STATUS_FILE = os.path.join(ROOT, "bot_runtime.json")
PID_FILE = os.path.join(ROOT, "bot_runtime.pid")
REGISTRY_PATH = os.path.join(MODELS_PATH, "registry.json")


ACTION_NAMES = {0: "FLAT", 1: "LONG", 2: "SHORT"}


def parse_args():
    p = argparse.ArgumentParser(description="Forex RL Bot — Live Trader (Demo Account)")
    p.add_argument("--pair", default="EURUSD",
                   help="Currency pair or 'all' for all pairs (default: EURUSD)")
    p.add_argument("--broker", default="paper", choices=["paper", "mt5"],
                   help="Broker backend (default: paper)")
    p.add_argument("--model", default=None, help="Path to model .zip (default: auto-detect)")
    p.add_argument("--use-best", action="store_true",
                   help="Pick best model per pair from models/registry.json (highest Sharpe)")
    p.add_argument("--units", type=int, default=LIVE_TRADE_UNITS,
                   help=f"Trade size in units (default: {LIVE_TRADE_UNITS})")
    p.add_argument("--dry-run", action="store_true", help="Show signal without placing orders")
    p.add_argument("--loop", action="store_true", help="Run continuously")
    p.add_argument("--interval", type=int, default=86400,
                   help="Loop interval in seconds (default 86400 = 24h)")
    p.add_argument("--status", action="store_true", help="Show paper account status and exit")
    p.add_argument("--reset", action="store_true", help="Reset paper account to $100k and exit")
    return p.parse_args()


# ── Status file helpers (consumed by the dashboard) ───────────────────────────

def write_status(state: str, **fields):
    """Write a small JSON file describing what the bot is doing right now."""
    payload = {
        "state": state,                          # idle / running / sleeping / stopped
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        **fields,
    }
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass  # don't crash the trader on a status-file error


def write_pid():
    """Write our PID so the dashboard can stop us cleanly."""
    try:
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass


def clear_runtime_files():
    for path in (STATUS_FILE, PID_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def create_client(broker: str):
    """Create the appropriate broker client."""
    if broker == "paper":
        from src.broker.paper_client import PaperClient
        return PaperClient()
    elif broker == "mt5":
        from src.broker.mt5_client import MT5Client
        return MT5Client()
    else:
        raise ValueError(f"Unknown broker: {broker}")


def get_symbol(pair: str, broker: str) -> str:
    """Convert pair name to the broker's symbol format."""
    # Paper and MT5 use plain names like "EURUSD"
    return pair


def _load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return None
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def _resolve_best_model_path(pair: str):
    """Look up the highest-Sharpe production model for `pair` in the registry.

    Production set: daily timeframe, v1 env, denoise=none|wavelet.
    Returns (path, agent_upper, denoise_method) or (None, None, None).
    """
    reg = _load_registry()
    if reg is None:
        return None, None, None
    candidates = [m for m in reg.get("models", [])
                  if m["pair"] == pair
                  and m.get("timeframe") == "1d"
                  and m.get("version", "v1") == "v1"
                  and m.get("denoise", "none") in ("none", "wavelet")]
    if not candidates:
        return None, None, None
    best = max(candidates, key=lambda m: m["sharpe_ratio"])
    den = best.get("denoise", "none")
    den_suffix = "" if den == "none" else f"_{den}"
    fname = f"{best['agent'].lower()}_feature_{pair}{den_suffix}.zip"
    path = os.path.join(MODELS_PATH, fname)
    if not os.path.exists(path):
        return None, None, None
    return path, best["agent"].upper(), den


def load_model(pair: str, model_path: str = None,
               use_best: bool = False, exit_on_fail: bool = True):
    """Load a trained model.

    Resolution order:
      1. If `model_path` is given, use it (DQN by default; PPO if filename starts with `ppo_`)
      2. If `use_best=True`, look up the registry's best model for this pair
      3. Otherwise default to models/dqn_feature_{PAIR}.zip
    """
    agent_upper = "DQN"
    self_denoise = "none"

    if model_path is None and use_best:
        resolved, agent_upper_resolved, den = _resolve_best_model_path(pair)
        if resolved is not None:
            model_path = resolved
            agent_upper = agent_upper_resolved
            self_denoise = den or "none"
            den_str = f"  (denoise={self_denoise})" if self_denoise != "none" else ""
            print(f"  Best model from registry: {os.path.basename(model_path)}{den_str}")

    if model_path is None:
        model_path = os.path.join(MODELS_PATH, f"dqn_feature_{pair}.zip")

    if not os.path.exists(model_path):
        msg = f"  Model not found: {model_path} — train with: python demo.py --pair {pair}"
        if exit_on_fail:
            print(f"ERROR: {msg}")
            sys.exit(1)
        else:
            print(f"  SKIP: {msg}")
            return None

    # Agent type from filename if not already set
    fname = os.path.basename(model_path).lower()
    if fname.startswith("ppo_"):
        agent_upper = "PPO"
    elif fname.startswith("dqn_"):
        agent_upper = "DQN"

    loader = PPO if agent_upper == "PPO" else DQN
    model = loader.load(model_path)
    print(f"  Model loaded ({agent_upper}): {model_path}")
    # Attach the denoise tag (set above when use_best) so callers can
    # preprocess data the same way the model was trained.
    model._denoise = self_denoise
    return model


def get_available_pairs() -> list:
    """Return pairs that have trained models."""
    available = []
    for pair in FOREX_PAIRS:
        path = os.path.join(MODELS_PATH, f"dqn_feature_{pair}.zip")
        if os.path.exists(path):
            available.append(pair)
    return available


def fetch_and_prepare(client, symbol: str, denoise: str = "none") -> pd.DataFrame:
    """Fetch live candles, compute indicators, and merge FRED economic data.

    If `denoise` is set, applies causal wavelet/EMD before indicators —
    must match how the loaded model was trained.
    """
    # Need 250+ candles: 200 for SMA_200 + buffer for indicator warm-up
    df = client.fetch_candles(symbol, count=300, granularity="D")
    print(f"  Fetched {len(df)} daily candles  "
          f"({df['Date'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df['Date'].iloc[-1].strftime('%Y-%m-%d')})")

    if denoise and denoise != "none":
        from src.features.denoising import denoise_ohlc_causal
        df = denoise_ohlc_causal(df, method=denoise, window=256, refresh_every=1)
        print(f"  Denoised OHLC with {denoise} (causal, window=256)")

    df = add_indicators(df)
    # Merge FRED economic features
    if os.path.exists(ECON_PATH):
        df = load_econ_features(ECON_PATH, df)
    # Add sentiment features
    df = add_sentiment_to_df(df, symbol, live=HAS_TEXTBLOB)
    print(f"  {len(df)} rows after indicators + econ + sentiment")
    return df


def get_model_action(model, df: pd.DataFrame) -> int:
    """
    Build the observation from the latest data and get the model's action.
    Uses the same logic as ForexFeatureEnv._get_obs().
    Auto-detects the model's expected observation size and adjusts features.
    """
    expected_dim = model.observation_space.shape[0]

    window = 20
    prices = df["Close"].values.astype(np.float64)
    idx = len(prices) - 1  # latest bar

    # Base observation: 20 returns + vol + position (assume flat for signal)
    window_prices = prices[idx - window: idx + 1]
    returns = np.diff(window_prices) / window_prices[:-1]
    rolling_vol = float(np.std(returns) * np.sqrt(252))
    position = 0.0  # assume flat for signal generation

    base_obs = np.append(returns, [rolling_vol, position]).astype(np.float32)

    # Feature observation: 7 indicators (same normalization as ForexFeatureEnv)
    close = float(prices[idx]) + 1e-8
    rsi = float(df["RSI"].iloc[-1]) / 100.0
    macd = float(df["MACD"].iloc[-1]) / close
    macd_sig = float(df["MACD_signal"].iloc[-1]) / close
    bb_width = float(df["BB_width"].iloc[-1])
    atr = float(df["ATR"].iloc[-1]) / close
    adx = float(df["ADX"].iloc[-1]) / 100.0
    sma_ratio = float(df["SMA_ratio"].iloc[-1]) - 1.0

    extra = [rsi, macd, macd_sig, bb_width, atr, adx, sma_ratio]

    # Append economic features if available
    if "rate_diff" in df.columns:
        extra.append(float(df["rate_diff"].iloc[-1]))
        extra.append(float(df["yield_curve"].iloc[-1]))
        extra.append(float(df["rate_diff_ma"].iloc[-1]))

    # Only include sentiment if the model was trained with it
    if "sentiment_score" in df.columns and (22 + len(extra) + 3) <= expected_dim:
        extra.append(float(df["sentiment_score"].iloc[-1]))
        extra.append(float(df["sentiment_vol"].iloc[-1]))
        extra.append(float(df["news_volume"].iloc[-1]))

    extra = np.array(extra, dtype=np.float32)
    obs = np.concatenate([base_obs, extra])

    # Pad or trim to match model's expected dimension
    if len(obs) < expected_dim:
        obs = np.concatenate([obs, np.zeros(expected_dim - len(obs), dtype=np.float32)])
    elif len(obs) > expected_dim:
        obs = obs[:expected_dim]

    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def execute_action(client, symbol: str, action: int,
                   units: int, dry_run: bool) -> None:
    """Map model action to broker orders."""
    current_pos = client.get_net_position(symbol)
    target_pos = {0: 0, 1: 1, 2: -1}[action]

    pos_names = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    print(f"  Current position: {pos_names[current_pos]}"
          f"  →  Target: {ACTION_NAMES[action]}")

    if target_pos == current_pos:
        print("  No change needed — holding current position.")
        return

    if dry_run:
        print("  [DRY RUN] Would execute the trade above. Use without --dry-run to trade.")
        return

    # Close existing position if switching sides
    if current_pos != 0:
        print("  Closing current position ...")
        client.close_position(symbol)

    # Open new position if not going flat
    if target_pos != 0:
        order_units = units if target_pos == 1 else -units
        print(f"  Placing order: {order_units:+d} units ...")
        result = client.place_order(symbol, order_units)
        print(f"  Filled at {result['price']} — PL: {result['pl']}")
    else:
        print("  Going flat — no new position.")


def log_trade(pair: str, action: int, balance: float, price: float):
    """Append trade decision to a CSV log."""
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_log.csv")
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "pair": pair,
        "action": ACTION_NAMES[action],
        "price": price,
        "balance": balance,
    }
    df = pd.DataFrame([entry])

    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

    print(f"  Logged to {log_path}")


def run_pair(pair: str, client, args):
    """Run one trading decision for a single pair."""
    symbol = get_symbol(pair, args.broker)

    print(f"\n  --- {pair} ---")

    # Load model (skip if missing in multi-pair mode)
    is_multi = args.pair.upper() == "ALL"
    model = load_model(pair, args.model, use_best=args.use_best,
                       exit_on_fail=not is_multi)
    if model is None:
        return

    # Fetch data and compute indicators (apply same denoising as training)
    model_denoise = getattr(model, "_denoise", "none")
    df = fetch_and_prepare(client, symbol, denoise=model_denoise)

    # Get model decision
    action = get_model_action(model, df)
    print(f"\n  Model signal: {ACTION_NAMES[action]}")

    # Execute
    execute_action(client, symbol, action, args.units, args.dry_run)

    # Log
    acct = client.get_account_summary()
    log_trade(pair, action, acct["balance"], df["Close"].iloc[-1])


def run_once(args):
    """Run one trading decision cycle for one or all pairs."""
    pair_input = args.pair.upper()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'=' * 55}")
    print(f"  Forex RL Bot — Live Trader  |  {now}")
    print(f"  Broker: {args.broker.upper()}")

    # Determine which pairs to trade
    if pair_input == "ALL":
        pairs = get_available_pairs()
        if not pairs:
            print("\n  ERROR: No trained models found in models/")
            print("  Train with:  python demo.py --pair EURUSD")
            sys.exit(1)
        print(f"  Pairs: {', '.join(pairs)}")
    else:
        pairs = [pair_input]
        print(f"  Pair: {pair_input}")

    print(f"{'=' * 55}")

    # Connect to broker
    client = create_client(args.broker)
    acct = client.get_account_summary()
    print(f"\n  Account balance: ${acct['balance']:,.2f}  "
          f"({acct['open_trades']} open trades)")

    # Trade each pair
    for pair in pairs:
        run_pair(pair, client, args)

    # Show paper account status after all trades
    if args.broker == "paper" and hasattr(client, "print_status"):
        client.print_status()

    # Cleanup MT5 connection
    if args.broker == "mt5" and hasattr(client, "shutdown"):
        client.shutdown()

    print()


def main():
    args = parse_args()

    # Handle --reset and --status for paper account
    if args.reset:
        from src.broker.paper_client import PaperClient
        client = PaperClient()
        client.reset_account()
        return

    if args.status:
        from src.broker.paper_client import PaperClient
        client = PaperClient()
        client.print_status()
        return

    if args.loop:
        write_pid()
        interval = max(60, int(args.interval))
        print(f"Running in loop mode every {interval}s (Ctrl+C to stop) ...")

        # Handle SIGTERM (e.g. from dashboard "Stop Bot" button) like Ctrl+C
        def _graceful_stop(signum, frame):
            print("\nReceived stop signal — shutting down cleanly.")
            write_status("stopped")
            clear_runtime_files()
            sys.exit(0)
        signal.signal(signal.SIGTERM, _graceful_stop)

        try:
            while True:
                cycle_start = datetime.datetime.now()
                write_status(
                    "running",
                    pair=args.pair,
                    broker=args.broker,
                    interval=interval,
                    cycle_started_at=cycle_start.isoformat(timespec="seconds"),
                )
                run_once(args)

                next_run = cycle_start + datetime.timedelta(seconds=interval)
                write_status(
                    "sleeping",
                    pair=args.pair,
                    broker=args.broker,
                    interval=interval,
                    last_cycle_at=cycle_start.isoformat(timespec="seconds"),
                    next_cycle_at=next_run.isoformat(timespec="seconds"),
                )
                print(f"  Sleeping {interval}s until next check (next: {next_run:%H:%M:%S}) ...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            write_status("stopped")
            clear_runtime_files()
    else:
        run_once(args)


if __name__ == "__main__":
    main()
