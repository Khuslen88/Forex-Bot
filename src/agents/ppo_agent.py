"""
ppo_agent.py — Train and run a PPO agent using Stable-Baselines3.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from src.environment.forex_env import ForexTradingEnv
from config.settings import PPO_CONFIG, PPO_NET_ARCH, MODELS_PATH, TOTAL_TIMESTEPS


def _linear_schedule(initial_lr: float):
    """Linear learning rate decay from initial_lr -> 10% of initial_lr."""
    def schedule(progress_remaining: float) -> float:
        return initial_lr * (0.1 + 0.9 * progress_remaining)
    return schedule


def train_ppo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_path: str = None,
    verbose: int = 1,
    env_class=ForexTradingEnv,
):
    """
    Train a PPO agent on a single-pair price DataFrame.

    Parameters
    ----------
    train_df        : DataFrame with a 'Close' column (training period)
    val_df          : Optional validation DataFrame — triggers EvalCallback
    total_timesteps : How many environment steps to train for
    save_path       : If set, saves the model to this path (.zip)
    verbose         : 0 = silent, 1 = progress output

    Returns
    -------
    (model, train_env)
    """
    train_env = env_class(train_df)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=_linear_schedule(PPO_CONFIG["learning_rate"]),
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        gamma=PPO_CONFIG["gamma"],
        clip_range=PPO_CONFIG["clip_range"],
        ent_coef=PPO_CONFIG["ent_coef"],
        max_grad_norm=PPO_CONFIG["max_grad_norm"],
        policy_kwargs={"net_arch": PPO_NET_ARCH},
        verbose=verbose,
    )

    callbacks = []
    if val_df is not None:
        val_env = env_class(val_df)
        eval_cb = EvalCallback(
            val_env,
            eval_freq=5_000,
            n_eval_episodes=1,
            verbose=0,
        )
        callbacks.append(eval_cb)

    model.learn(total_timesteps=total_timesteps, callback=callbacks or None)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        if verbose:
            print(f"Model saved -> {save_path}")

    return model, train_env


def run_ppo(model, df: pd.DataFrame, env_class=ForexTradingEnv):
    """
    Run a trained PPO model on a DataFrame and collect the equity curve + trades.

    Parameters
    ----------
    model : trained Stable-Baselines3 PPO model
    df    : test-period DataFrame with a 'Close' column

    Returns
    -------
    (equity_curve: list[float], trades: list[dict])
    """
    env   = env_class(df)
    obs, _ = env.reset()

    equity_curve  = [env.initial_balance]
    trades        = []
    prev_position = 0
    entry_price   = None
    entry_balance = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))

        equity_curve.append(info["balance"])
        new_position = info["position"]

        if new_position != prev_position:
            if prev_position != 0 and entry_price is not None:
                trades.append({
                    "entry":     entry_price,
                    "exit":      df["Close"].iloc[min(env.current_step - 1, len(df) - 1)],
                    "direction": "long" if prev_position == 1 else "short",
                    "pnl":       info["balance"] - entry_balance,
                })
            if new_position != 0:
                entry_price   = df["Close"].iloc[min(env.current_step - 1, len(df) - 1)]
                entry_balance = info["balance"]

        prev_position = new_position

        if terminated or truncated:
            break

    # Close any still-open position at end of episode
    if prev_position != 0 and entry_price is not None:
        trades.append({
            "entry":     entry_price,
            "exit":      df["Close"].iloc[-1],
            "direction": "long" if prev_position == 1 else "short",
            "pnl":       equity_curve[-1] - entry_balance,
        })

    return equity_curve, trades


def load_ppo(path: str, df: pd.DataFrame, env_class=ForexTradingEnv):
    """Load a saved PPO model (.zip) and bind it to a fresh environment."""
    env   = env_class(df)
    model = PPO.load(path, env=env)
    return model
