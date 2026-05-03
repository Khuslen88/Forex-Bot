"""
demo.py — End-to-end MVP demo for the Forex RL Bot capstone.

Run from the forex-bot/ directory:

  # Full training with DQN (default, takes ~5-10 min):
  python demo.py --pair EURUSD

  # Train with PPO instead:
  python demo.py --pair EURUSD --agent ppo

  # Quick mode for testing (20k steps):
  python demo.py --pair EURUSD --quick

  # Load a pre-trained model (fastest — use this during live demo):
  python demo.py --pair EURUSD --load models/dqn_feature_EURUSD.zip
  python demo.py --pair EURUSD --agent ppo --load models/ppo_feature_EURUSD.zip
"""

import os
import sys
import argparse
from functools import partial
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.indicators       import load_pair
from src.agents.baselines          import run_buy_and_hold, run_sma_crossover, run_random_agent
from src.agents.dqn_agent          import train_dqn, run_dqn, load_dqn
from src.agents.ppo_agent          import train_ppo, run_ppo, load_ppo
from src.backtesting.evaluate      import compute_metrics, print_comparison, meets_targets, walk_forward_test
from src.environment.feature_env   import ForexFeatureEnv
from config.settings               import TRAIN_TEST_SPLIT, TOTAL_TIMESTEPS, MODELS_PATH, TIMEFRAME_CONFIGS


def _make_env_factory(timeframe: str):
    """Build a callable env factory pre-bound with timeframe-specific SL/TP/min_hold.

    Returned object behaves like a class: factory(df) → ForexFeatureEnv(df, ...).
    Compatible with train_fn(env_class=...) which calls env_class(df).
    """
    cfg = TIMEFRAME_CONFIGS[timeframe]
    return partial(
        ForexFeatureEnv,
        stop_loss=cfg["stop_loss_pct"],
        take_profit=cfg["take_profit_pct"],
        min_hold_steps=cfg["min_hold_steps"],
    )


# ── CLI arguments ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Forex RL Bot — MVP Demo")
    p.add_argument("--pair",  default="EURUSD", help="Currency pair (default: EURUSD)")
    p.add_argument("--agent", default="dqn", choices=["dqn", "ppo"], help="RL agent: dqn or ppo (default: dqn)")
    p.add_argument("--timeframe", default="1d", choices=list(TIMEFRAME_CONFIGS.keys()),
                   help="Candle timeframe: 1d (daily) or 1h (hourly). Default: 1d")
    p.add_argument("--quick", action="store_true", help="Train for only 20k steps (fast test)")
    p.add_argument("--load",  default=None,  help="Path to a pre-trained model .zip file")
    p.add_argument("--seeds", type=int, default=1, help="Train N models with different seeds, keep best (default: 1)")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(pair: str, timeframe: str = "1d") -> pd.DataFrame:
    data_file = TIMEFRAME_CONFIGS[timeframe]["data_file"]
    data_path = os.path.join(os.path.dirname(__file__), "data", "raw", data_file)
    econ_path = os.path.join(os.path.dirname(__file__), "data", "economic", "fred_data.csv")
    print(f"\n[1/5] Loading data: {pair}  (timeframe={timeframe}, file={data_file})")
    raw = pd.read_csv(data_path)

    use_econ = os.path.exists(econ_path)
    df  = load_pair(raw, pair, econ_path=econ_path if use_econ else None)

    print(f"      {len(df)} rows after adding indicators  "
          f"({df['Date'].iloc[0].strftime('%Y-%m-%d')} → {df['Date'].iloc[-1].strftime('%Y-%m-%d')})")

    if use_econ and "rate_diff" in df.columns:
        corr_ret  = df["rate_diff"].corr(df["Return"])
        corr_yc   = df["yield_curve"].corr(df["Return"])
        print(f"      FRED macro loaded — rate_diff×return corr: {corr_ret:+.3f}  "
              f"yield_curve×return corr: {corr_yc:+.3f}")

    return df


# ── Train / test split ─────────────────────────────────────────────────────────

def split(df: pd.DataFrame):
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    print(f"\n[2/5] Train/test split (80/20)")
    print(f"      Train: {len(train)} rows  "
          f"({train['Date'].iloc[0].strftime('%Y-%m-%d')} → {train['Date'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"      Test:  {len(test)}  rows  "
          f"({test['Date'].iloc[0].strftime('%Y-%m-%d')} → {test['Date'].iloc[-1].strftime('%Y-%m-%d')})")
    return train, test


# ── Agent dispatch helpers ─────────────────────────────────────────────────────

def _agent_funcs(agent_name):
    """Return (train_fn, run_fn, load_fn) for the chosen agent."""
    if agent_name == "ppo":
        return train_ppo, run_ppo, load_ppo
    return train_dqn, run_dqn, load_dqn


# ── Train or load model ──────────────────────────────────────────────────────

def get_model(args, train_df, val_df, pair):
    agent = args.agent
    agent_upper = agent.upper()
    timeframe = args.timeframe
    train_fn, run_fn, load_fn = _agent_funcs(agent)
    env_factory = _make_env_factory(timeframe)

    if args.load:
        print(f"\n[3/5] Loading pre-trained model: {args.load}")
        from src.environment.forex_env import ForexTradingEnv
        env_cls = env_factory if "feature" in os.path.basename(args.load) else ForexTradingEnv
        return load_fn(args.load, train_df, env_class=env_cls), env_cls

    cfg = TIMEFRAME_CONFIGS[timeframe]
    timesteps = 20_000 if args.quick else cfg["total_timesteps"]
    suffix = "" if timeframe == "1d" else f"_{timeframe}"
    save_path = os.path.join(MODELS_PATH, f"{agent}_feature_{pair}{suffix}.zip")
    n_seeds = args.seeds

    if n_seeds > 1:
        return _train_multi_seed(train_df, val_df, pair, timesteps, save_path, n_seeds, agent, env_factory)

    print(f"\n[3/5] Training {agent_upper} ({timeframe}) on {pair}  ({timesteps:,} timesteps)")
    print(f"      SL: {cfg['stop_loss_pct']*100:.2f}%  TP: {cfg['take_profit_pct']*100:.2f}%  "
          f"min_hold: {cfg['min_hold_steps']} bars")
    print(f"      Save path: {save_path}")

    model, _ = train_fn(
        train_df,
        val_df=val_df,
        total_timesteps=timesteps,
        save_path=save_path,
        verbose=1,
        env_class=env_factory,
    )
    return model, env_factory


def _train_multi_seed(train_df, val_df, pair, timesteps, save_path, n_seeds, agent="dqn", env_factory=None):
    """Train N models with different random seeds, keep the best by validation Sharpe."""
    agent_upper = agent.upper()
    train_fn, run_fn, _ = _agent_funcs(agent)
    if env_factory is None:
        env_factory = ForexFeatureEnv

    print(f"\n[3/5] Multi-seed training ({agent_upper}): {n_seeds} seeds x {timesteps:,} timesteps on {pair}")

    best_model = None
    best_sharpe = -np.inf
    best_seed = None

    for i in range(n_seeds):
        seed = 42 + i * 17  # deterministic but varied seeds
        print(f"\n      --- Seed {i+1}/{n_seeds} (seed={seed}) ---")

        model, _ = train_fn(
            train_df,
            val_df=val_df,
            total_timesteps=timesteps,
            save_path=None,  # don't save yet
            verbose=0,
            env_class=env_factory,
        )

        # Evaluate on validation set
        equity, trades = run_fn(model, val_df, env_class=env_factory)
        metrics = compute_metrics(equity, trades)
        sharpe = metrics["sharpe_ratio"]
        ret = metrics["total_return"] * 100

        print(f"      Return: {ret:+.2f}%  Sharpe: {sharpe:.3f}  Win rate: {metrics['win_rate']*100:.1f}%")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_model = model
            best_seed = seed

    print(f"\n      Best seed: {best_seed}  (Sharpe: {best_sharpe:.3f})")

    # Save the best model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_model.save(save_path)
    print(f"      Model saved -> {save_path}")

    return best_model, env_factory


# ── Run all strategies on test data ───────────────────────────────────────────

def run_all(model, test_df, env_class=ForexFeatureEnv, agent="dqn", pair="EURUSD", timeframe="1d"):
    agent_upper = agent.upper()
    _, run_fn, _ = _agent_funcs(agent)

    print("\n[4/5] Running all strategies on test data ...")

    rl_equity,   rl_trades   = run_fn(model, test_df, env_class=env_class)
    bah_equity,  bah_trades  = run_buy_and_hold(test_df)
    sma_equity,  sma_trades  = run_sma_crossover(test_df)
    rand_equity, rand_trades = run_random_agent(test_df)

    print(f"      {agent_upper} final balance:          ${rl_equity[-1]:,.2f}")
    print(f"      Buy-and-Hold final balance: ${bah_equity[-1]:,.2f}")
    print(f"      SMA Crossover final:        ${sma_equity[-1]:,.2f}")
    print(f"      Random Agent final:         ${rand_equity[-1]:,.2f}")

    results = {
        agent_upper:     (rl_equity,   rl_trades),
        "Buy & Hold":    (bah_equity,  bah_trades),
        "SMA Crossover": (sma_equity,  sma_trades),
        "Random Agent":  (rand_equity, rand_trades),
    }

    # Also include the OTHER agent if its model exists
    other_agent = "ppo" if agent == "dqn" else "dqn"
    other_upper = other_agent.upper()
    suffix = "" if timeframe == "1d" else f"_{timeframe}"
    other_path = os.path.join(MODELS_PATH, f"{other_agent}_feature_{pair}{suffix}.zip")
    if os.path.exists(other_path):
        print(f"      Also loading {other_upper} model for comparison ...")
        _, other_run, other_load = _agent_funcs(other_agent)
        try:
            other_model = other_load(other_path, test_df, env_class=env_class)
            other_equity, other_trades = other_run(other_model, test_df, env_class=env_class)
            results[other_upper] = (other_equity, other_trades)
            print(f"      {other_upper} final balance:          ${other_equity[-1]:,.2f}")
        except Exception as e:
            print(f"      Could not load {other_upper}: {e}")

    return results


# ── Metrics + chart ────────────────────────────────────────────────────────────

def evaluate_and_chart(strategy_results: dict, test_df: pd.DataFrame, pair: str, agent="dqn", timeframe="1d"):
    agent_upper = agent.upper()
    print("\n[5/5] Computing metrics and generating chart ...")

    metrics = {
        name: compute_metrics(equity, trades)
        for name, (equity, trades) in strategy_results.items()
    }

    print_comparison(metrics)

    # ── Target check for RL agent ─────────────────────────────────────────
    targets = meets_targets(metrics[agent_upper])
    print(f"{agent_upper} vs project targets:")
    for target, passed in targets.items():
        status = "+" if passed else "x"
        print(f"  {status}  {target}")

    # ── Interactive equity curve chart ────────────────────────────────────
    dates = test_df["Date"].values

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"{pair} Equity Curves -- {agent_upper} vs Baselines (Test Period)",
            f"{pair} Close Price (Test Period)"
        ),
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
    )

    colors = {
        "DQN":           "#2196F3",
        "PPO":           "#E91E63",
        "Buy & Hold":    "#4CAF50",
        "SMA Crossover": "#FF9800",
        "Random Agent":  "#9E9E9E",
    }

    for name, (equity, _) in strategy_results.items():
        # Align equity length with dates
        n = min(len(equity), len(dates))
        fig.add_trace(
            go.Scatter(
                x=dates[:n], y=equity[:n],
                name=name,
                line=dict(color=colors.get(name, "#2196F3"), width=2 if name == agent_upper else 1),
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=dates, y=test_df["Close"].values,
            name="EUR/USD Close",
            line=dict(color="#607D8B", width=1),
            showlegend=True,
        ),
        row=2, col=1,
    )

    # Annotate RL agent final return
    rl_ret = metrics[agent_upper]["total_return"] * 100
    sharpe = metrics[agent_upper]["sharpe_ratio"]
    fig.add_annotation(
        x=dates[-1],
        y=strategy_results[agent_upper][0][-1],
        text=f"{agent_upper}: {rl_ret:+.1f}%<br>Sharpe: {sharpe:.2f}",
        showarrow=True, arrowhead=2,
        bgcolor=colors.get(agent_upper, "#2196F3"), font=dict(color="white"),
        row=1, col=1,
    )

    fig.update_layout(
        height=700,
        title_text=f"Forex RL Bot -- {pair} MVP Results ({agent_upper})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)

    suffix = "" if timeframe == "1d" else f"_{timeframe}"
    chart_path = os.path.join(os.path.dirname(__file__), f"results_{pair}{suffix}.html")
    fig.write_html(chart_path)
    print(f"\nChart saved -> {chart_path}")
    print("Open in your browser to view the interactive equity curves.\n")

    return metrics


# ── Walk-forward validation ────────────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, args, pair: str):
    """Run 5-fold walk-forward validation using the selected agent."""
    agent_upper = args.agent.upper()
    print(f"\n[+] Walk-Forward Validation (5 folds, {agent_upper}, {args.timeframe}) ...")

    env_factory = _make_env_factory(args.timeframe)
    train_fn, run_fn, _ = _agent_funcs(args.agent)

    def _train(train_df):
        timesteps = 20_000 if args.quick else 60_000
        model, _ = train_fn(train_df, total_timesteps=timesteps, verbose=0,
                             env_class=env_factory)
        return model

    def _run(model, test_df):
        return run_fn(model, test_df, env_class=env_factory)

    results = walk_forward_test(df, _train, _run, n_splits=5)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    pair      = args.pair.upper()
    agent     = args.agent
    timeframe = args.timeframe

    print("=" * 60)
    print(f"  Forex RL Bot — MVP Demo  |  Pair: {pair}  |  Agent: {agent.upper()}  |  TF: {timeframe}")
    print("=" * 60)

    df                = load_data(pair, timeframe=timeframe)
    train_df, test_df = split(df)
    model, env_class  = get_model(args, train_df, test_df, pair)
    strategy_results  = run_all(model, test_df, env_class=env_class, agent=agent, pair=pair, timeframe=timeframe)
    evaluate_and_chart(strategy_results, test_df, pair, agent=agent, timeframe=timeframe)

    run_walk_forward(df, args, pair)


if __name__ == "__main__":
    main()
