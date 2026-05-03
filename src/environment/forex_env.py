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
    df              : DataFrame with at least a 'Close' column, indexed by date.
                      For ATR-based SL/TP, must also contain 'ATR'.
    window          : Number of past bar returns fed into the observation vector.
    initial_balance : Starting account balance in USD.
    trading_cost    : Flat per-trade cost (legacy). Used only when spread_pips==0.
    stop_loss       : Fixed-pct SL (legacy). Used only when atr_mult_sl==0.
    take_profit     : Fixed-pct TP (legacy). Used only when atr_mult_tp==0.
    min_hold_steps  : Block agent from changing position before holding this long.
    atr_mult_sl     : v2 — if > 0, SL distance = atr_mult_sl * ATR_at_entry.
                      Overrides stop_loss. Adapts to current volatility.
    atr_mult_tp     : v2 — if > 0, TP distance = atr_mult_tp * ATR_at_entry.
                      Overrides take_profit.
    spread_pips     : v2 — if > 0, charge variable cost = (spread + slippage)
                      pips per position change. Overrides trading_cost.
    slippage_pips   : v2 — extra pips of slippage on top of spread.
    pip_size        : v2 — pip unit in price (0.0001 for most, 0.01 for JPY).
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
        min_hold_steps: int = 0,
        atr_mult_sl: float = 0.0,
        atr_mult_tp: float = 0.0,
        spread_pips: float = 0.0,
        slippage_pips: float = 0.0,
        pip_size: float = 0.0001,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window = window
        self.initial_balance = initial_balance
        self.trading_cost = trading_cost
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.min_hold_steps = min_hold_steps

        # v2 dynamic SL/TP + realistic cost model
        self.atr_mult_sl   = atr_mult_sl
        self.atr_mult_tp   = atr_mult_tp
        self.spread_pips   = spread_pips
        self.slippage_pips = slippage_pips
        self.pip_size      = pip_size

        self.use_atr_sl_tp = (atr_mult_sl > 0 or atr_mult_tp > 0)
        self.use_var_cost  = (spread_pips > 0)

        if self.use_atr_sl_tp and "ATR" not in self.df.columns:
            raise ValueError(
                "ATR-based SL/TP requested but 'ATR' column missing from df. "
                "Run add_indicators() first or use ForexFeatureEnv."
            )

        self.prices = self.df["Close"].values.astype(np.float64)
        self.atrs   = (self.df["ATR"].values.astype(np.float64)
                       if "ATR" in self.df.columns else None)

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

    def _trade_cost_fraction(self, price: float) -> float:
        """Per-position-change cost expressed as a fraction of price.

        v2: variable spread + slippage in pips, scaled by pip_size / price.
        v1 (legacy): flat self.trading_cost.
        """
        if self.use_var_cost:
            total_pips = self.spread_pips + self.slippage_pips
            return (total_pips * self.pip_size) / max(price, 1e-9)
        return self.trading_cost

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
        self.entry_atr       = None                 # ATR at entry — locks in SL/TP distance

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Execute one trading bar.

        action : int — 0 = Flat, 1 = Long, 2 = Short
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        sl_triggered = False
        tp_triggered = False

        # Map discrete action → position target
        target_position = {0: 0, 1: 1, 2: -1}[action]

        # Enforce minimum hold duration — block position changes too soon
        # (SL/TP can still close positions; this only restricts agent decisions)
        if (
            self.position != 0
            and target_position != self.position
            and self.steps_in_position < self.min_hold_steps
        ):
            target_position = self.position  # force-hold

        # Detect position change
        position_changed = target_position != self.position

        # Apply trading cost only when the position actually changes
        cost = (self._trade_cost_fraction(self.prices[self.current_step])
                if position_changed else 0.0)

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

        # Record entry price + ATR when opening a new position from flat
        prev_position = self.position
        self.position = target_position

        if position_changed and prev_position == 0 and self.position != 0:
            self.entry_price = self.prices[self.current_step]
            if self.atrs is not None:
                self.entry_atr = float(self.atrs[self.current_step])
            else:
                self.entry_atr = None
        elif position_changed and self.position == 0:
            self.entry_price = None
            self.entry_atr = None

        # Bar return: close[t+1] / close[t] - 1
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
            # Compute SL/TP distances — ATR-scaled (v2) or fixed-pct (v1)
            if self.use_atr_sl_tp and self.entry_atr is not None and self.entry_atr > 0:
                sl_dist = self.atr_mult_sl * self.entry_atr if self.atr_mult_sl > 0 else \
                          self.entry_price * self.stop_loss
                tp_dist = self.atr_mult_tp * self.entry_atr if self.atr_mult_tp > 0 else \
                          self.entry_price * self.take_profit
            else:
                sl_dist = self.entry_price * self.stop_loss
                tp_dist = self.entry_price * self.take_profit

            if self.position == 1:  # LONG
                if current_price <= self.entry_price - sl_dist:
                    sl_triggered = True
                elif current_price >= self.entry_price + tp_dist:
                    tp_triggered = True
            elif self.position == -1:  # SHORT
                if current_price >= self.entry_price + sl_dist:
                    sl_triggered = True
                elif current_price <= self.entry_price - tp_dist:
                    tp_triggered = True

            if sl_triggered or tp_triggered:
                self.position = 0
                self.entry_price = None
                self.entry_atr = None

        # Update peak for drawdown tracking
        self.peak_balance = max(self.peak_balance, self.balance)

        # ── Reward: risk-adjusted PnL ────────────────────────────────────
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
