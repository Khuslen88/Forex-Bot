"""
forex_env.py — Custom Gymnasium trading environment for the Forex RL Bot.

State  : window of daily returns + rolling volatility + current position
Actions: 0 = Flat  |  1 = Long  |  2 = Short
Reward : Risk-adjusted PnL with drawdown penalty and position-change cost
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config.settings import STOP_LOSS_PCT, TAKE_PROFIT_PCT


class ForexTradingEnv(gym.Env):
    """
    Single-pair forex trading environment compatible with Stable-Baselines3.

    Parameters
    ----------
    df           : DataFrame with at least a 'Close' column, indexed by date.
    window       : Number of past daily returns fed into the observation vector.
    initial_balance : Starting account balance in USD.
    trading_cost : One-way cost per trade as a fraction of price (default 1 pip).
    stop_loss    : Fraction of entry price for stop-loss trigger (default from settings).
    take_profit  : Fraction of entry price for take-profit trigger (default from settings).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 20,
        initial_balance: float = 10_000.0,
        trading_cost: float = 0.0001,
        stop_loss: float = STOP_LOSS_PCT,
        take_profit: float = TAKE_PROFIT_PCT,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window = window
        self.initial_balance = initial_balance
        self.trading_cost = trading_cost
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.prices = self.df["Close"].values.astype(np.float64)

        # ── Action space: Flat / Long / Short ─────────────────────────────
        self.action_space = spaces.Discrete(3)

        # ── Observation space ─────────────────────────────────────────────
        # [ returns_t-window … returns_t-1 ] + [ rolling_vol ] + [ position ]
        obs_size = window + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector for the current step."""
        window_prices = self.prices[self.current_step - self.window: self.current_step + 1]
        returns = np.diff(window_prices) / window_prices[:-1]          # shape: (window,)
        rolling_vol = float(np.std(returns) * np.sqrt(252))            # annualised
        obs = np.append(returns, [rolling_vol, float(self.position)])
        return obs.astype(np.float32)

    # ── Core Gym interface ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step    = self.window          # first valid step (needs window history)
        self.position        = 0                    # -1 = short | 0 = flat | 1 = long
        self.balance         = self.initial_balance
        self.peak_balance    = self.initial_balance
        self.equity_curve    = [self.initial_balance]
        self.returns_history = []
        self.steps_in_position = 0
        self.position_entry_balance = self.initial_balance
        self.entry_price     = None                 # price when position was opened

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Execute one trading day.

        action : int — 0 = Flat, 1 = Long, 2 = Short
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        sl_triggered = False
        tp_triggered = False

        # Map discrete action → position target
        target_position = {0: 0, 1: 1, 2: -1}[action]

        # Detect position change
        position_changed = target_position != self.position

        # Apply trading cost only when the position actually changes
        cost = self.trading_cost if position_changed else 0.0

        # Track trade completion for bonus/penalty
        trade_pnl = 0.0
        if position_changed and self.position != 0:
            # Closing a trade — calculate trade result
            trade_pnl = (self.balance - self.position_entry_balance) / self.position_entry_balance

        if position_changed:
            self.steps_in_position = 0
            self.position_entry_balance = self.balance
        else:
            self.steps_in_position += 1

        # Record entry price when opening a new position from flat
        prev_position = self.position
        self.position = target_position

        if position_changed and prev_position == 0 and self.position != 0:
            self.entry_price = self.prices[self.current_step]
        elif position_changed and self.position == 0:
            self.entry_price = None

        # Daily return: close[t+1] / close[t] - 1
        daily_return = (
            self.prices[self.current_step + 1] - self.prices[self.current_step]
        ) / self.prices[self.current_step]

        # PnL for this step (+return if long, -return if short, −cost either way)
        pnl = self.position * daily_return - cost
        self.balance *= 1.0 + pnl
        self.equity_curve.append(self.balance)
        self.returns_history.append(pnl)

        # ── Stop Loss / Take Profit check ──────────────────────────────
        current_price = self.prices[self.current_step + 1]

        if self.position != 0 and self.entry_price is not None:
            if self.position == 1:  # LONG
                if current_price <= self.entry_price * (1.0 - self.stop_loss):
                    sl_triggered = True
                elif current_price >= self.entry_price * (1.0 + self.take_profit):
                    tp_triggered = True
            elif self.position == -1:  # SHORT
                if current_price >= self.entry_price * (1.0 + self.stop_loss):
                    sl_triggered = True
                elif current_price <= self.entry_price * (1.0 - self.take_profit):
                    tp_triggered = True

            if sl_triggered or tp_triggered:
                self.position = 0
                self.entry_price = None

        # Update peak for drawdown tracking
        self.peak_balance = max(self.peak_balance, self.balance)

        # ── Reward: risk-adjusted PnL ────────────────────────────────────
        #
        # Components:
        #   1. Risk-adjusted PnL: divide by rolling volatility so the agent
        #      learns to trade relative to current market conditions
        #   2. Drawdown penalty: penalise being far from peak equity
        #   3. Trade completion bonus: reward closing profitable trades
        #   4. Mild flat penalty: discourage sitting out entirely
        #   5. SL/TP reward shaping: bonus for TP, mild penalty for SL

        # 1. Risk-adjusted PnL (Sharpe-like per step)
        if len(self.returns_history) >= 5:
            recent_vol = float(np.std(self.returns_history[-20:]))
        else:
            recent_vol = 0.01  # default early in episode
        risk_adj_pnl = pnl / (recent_vol + 1e-6)
        reward = float(np.clip(risk_adj_pnl, -3.0, 3.0))

        # 2. Drawdown penalty — penalise being in drawdown
        current_dd = (self.balance - self.peak_balance) / self.peak_balance
        if current_dd < -0.05:  # only penalise beyond 5% drawdown
            reward += current_dd * 2.0  # e.g. -10% dd → -0.2 penalty

        # 3. Trade completion bonus — reward closing profitable trades
        if position_changed and trade_pnl != 0:
            if trade_pnl > 0:
                reward += min(trade_pnl * 50, 1.0)   # cap bonus at 1.0
            else:
                reward += max(trade_pnl * 30, -0.5)   # cap loss penalty at -0.5

        # 4. Mild flat penalty — only after being flat for >5 steps
        if self.position == 0 and self.steps_in_position > 5:
            reward -= 0.05

        # 5. SL/TP reward shaping
        if tp_triggered:
            reward += 0.5    # bonus for hitting take profit
        elif sl_triggered:
            reward -= 0.3    # mild penalty for stop loss (protective, so less harsh)

        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        truncated  = False

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        info = {
            "balance"      : self.balance,
            "position"     : self.position,
            "daily_return" : daily_return,
            "pnl"          : pnl,
            "sl_triggered" : sl_triggered,
            "tp_triggered" : tp_triggered,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        step  = self.current_step
        pos   = {1: "LONG", -1: "SHORT", 0: "FLAT"}[self.position]
        pct   = (self.balance / self.initial_balance - 1) * 100
        print(f"[Step {step:4d}]  {pos:5s}  Balance: ${self.balance:,.2f}  ({pct:+.2f}%)")
