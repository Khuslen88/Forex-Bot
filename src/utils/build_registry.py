"""
build_registry.py — Compute test-set metrics for every saved model and write
them to models/registry.json. The dashboard reads this file to display
"best model per pair" recommendations and historical performance.

Run after training to refresh the registry:
    python src/utils/build_registry.py
"""

import os
import sys
import json
import glob
from datetime import datetime
from functools import partial

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.features.indicators       import load_pair
from src.environment.feature_env   import ForexFeatureEnv
from src.agents.dqn_agent          import load_dqn, run_dqn
from src.agents.ppo_agent          import load_ppo, run_ppo
from src.backtesting.evaluate      import compute_metrics
from config.settings               import TRAIN_TEST_SPLIT, TIMEFRAME_CONFIGS, MODELS_PATH


REGISTRY_PATH = os.path.join(MODELS_PATH, "registry.json")
DATA_DIR      = os.path.join(ROOT, "data", "raw")
ECON_PATH     = os.path.join(ROOT, "data", "economic", "fred_data.csv")


def parse_model_filename(filename: str):
    """Extract (agent, pair, timeframe) from a model filename.

    Examples:
        dqn_feature_EURUSD.zip      -> ("dqn", "EURUSD", "1d")
        ppo_feature_USDCAD_1h.zip   -> ("ppo", "USDCAD", "1h")
    """
    name = os.path.basename(filename).replace(".zip", "")
    parts = name.split("_")
    if len(parts) < 3:
        return None
    agent = parts[0]
    if agent not in ("dqn", "ppo"):
        return None
    if parts[1] != "feature":
        return None
    pair = parts[2]
    timeframe = parts[3] if len(parts) >= 4 else "1d"
    if timeframe not in TIMEFRAME_CONFIGS:
        return None
    return agent, pair, timeframe


def load_test_df(pair: str, timeframe: str) -> pd.DataFrame:
    cfg = TIMEFRAME_CONFIGS[timeframe]
    raw = pd.read_csv(os.path.join(DATA_DIR, cfg["data_file"]))
    df = load_pair(raw, pair, econ_path=ECON_PATH if os.path.exists(ECON_PATH) else None)
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def make_env_factory(timeframe: str):
    cfg = TIMEFRAME_CONFIGS[timeframe]
    return partial(
        ForexFeatureEnv,
        stop_loss=cfg["stop_loss_pct"],
        take_profit=cfg["take_profit_pct"],
        min_hold_steps=cfg["min_hold_steps"],
    )


def evaluate_model(model_path: str):
    parsed = parse_model_filename(model_path)
    if parsed is None:
        return None
    agent, pair, timeframe = parsed

    test_df = load_test_df(pair, timeframe)
    env_factory = make_env_factory(timeframe)

    if agent == "dqn":
        model = load_dqn(model_path, test_df, env_class=env_factory)
        equity, trades = run_dqn(model, test_df, env_class=env_factory)
    else:
        model = load_ppo(model_path, test_df, env_class=env_factory)
        equity, trades = run_ppo(model, test_df, env_class=env_factory)

    metrics = compute_metrics(equity, trades)
    metrics["agent"]        = agent.upper()
    metrics["pair"]         = pair
    metrics["timeframe"]    = timeframe
    metrics["model_path"]   = os.path.relpath(model_path, ROOT)
    metrics["model_size_kb"] = round(os.path.getsize(model_path) / 1024, 1)
    metrics["computed_at"]  = datetime.now().isoformat(timespec="seconds")
    metrics["n_test_rows"]  = len(test_df)
    metrics["n_trades"]     = len(trades)
    metrics["final_balance"] = float(equity[-1])
    return metrics


def main():
    pattern = os.path.join(MODELS_PATH, "*_feature_*.zip")
    models = sorted(glob.glob(pattern))
    print(f"Found {len(models)} model(s) to evaluate.\n")

    registry = {"models": [], "updated_at": datetime.now().isoformat(timespec="seconds")}

    for path in models:
        rel = os.path.relpath(path, ROOT)
        print(f"  {rel} ...", end=" ", flush=True)
        try:
            metrics = evaluate_model(path)
            if metrics is None:
                print("skipped (unparseable name)")
                continue
            print(f"Sharpe {metrics['sharpe_ratio']:+.3f}  "
                  f"Return {metrics['total_return']*100:+.2f}%  "
                  f"Win {metrics['win_rate']*100:.1f}%")
            registry["models"].append(metrics)
        except Exception as e:
            print(f"FAILED: {e}")

    # Index by best agent per pair (per timeframe), ranked by Sharpe
    best_by_pair = {}
    for m in registry["models"]:
        key = f"{m['pair']}_{m['timeframe']}"
        if key not in best_by_pair or m["sharpe_ratio"] > best_by_pair[key]["sharpe_ratio"]:
            best_by_pair[key] = m
    registry["best_by_pair"] = best_by_pair

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nSaved registry -> {REGISTRY_PATH}")
    print(f"Total models indexed: {len(registry['models'])}")


if __name__ == "__main__":
    main()
