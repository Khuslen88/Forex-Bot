"""
feature_env.py — Enhanced trading environment that adds technical indicators,
macroeconomic features, and sentiment to the observation vector.

Extends ForexTradingEnv by appending normalized indicator values to the state:
  RSI, MACD, MACD_signal, BB_width, ATR, ADX, SMA_ratio
  + optional: rate_diff, yield_curve, rate_diff_ma
  + optional: sentiment_score, sentiment_vol, news_volume

Requires the DataFrame to have been processed by src/features/indicators.py.

Observation vector (size = window + 2 + N_EXTRA):
  [ daily_returns(20), rolling_vol, position,
    rsi, macd, macd_sig, bb_width, atr, adx, sma_ratio,
    rate_diff, yield_curve, rate_diff_ma,              ← if available
    sentiment_score, sentiment_vol, news_volume ]      ← if available
"""

import numpy as np
from gymnasium import spaces
from src.environment.forex_env import ForexTradingEnv
from config.settings import STOP_LOSS_PCT, TAKE_PROFIT_PCT


class ForexFeatureEnv(ForexTradingEnv):
    """
    ForexTradingEnv + technical indicator and macro features in the observation.

    All indicators are normalized to be roughly in the same scale so the
    neural network can learn from them without one feature dominating.

    Normalization:
      RSI          → divide by 100          (maps 0–100 to 0–1)
      MACD         → divide by Close price  (makes it scale-invariant)
      MACD_signal  → divide by Close price
      BB_width     → already a ratio        (e.g. 0.01–0.05)
      ATR          → divide by Close price  (makes it scale-invariant)
      ADX          → divide by 100          (maps 0–100 to 0–1; >0.25 = trending)
      SMA_ratio    → subtract 1             (centers around 0; +0.02 = uptrend)
      rate_diff    → already small scale    (~-3 to +3)
      yield_curve  → already small scale    (~-1 to +3)
      rate_diff_ma → already small scale
    """

    # Required indicators (always present)
    INDICATOR_COLS = ["RSI", "MACD", "MACD_signal", "BB_width", "ATR", "ADX", "SMA_ratio"]

    # Optional macro features (present if FRED data was merged)
    ECON_COLS = ["rate_diff", "yield_curve", "rate_diff_ma"]

    # Optional sentiment features
    SENTIMENT_COLS = ["sentiment_score", "sentiment_vol", "news_volume"]

    def __init__(
        self,
        df,
        window: int = 20,
        initial_balance: float = 10_000.0,
        trading_cost: float = 0.0001,
        stop_loss: float = STOP_LOSS_PCT,
        take_profit: float = TAKE_PROFIT_PCT,
        min_hold_steps: int = 0,
    ):
        super().__init__(
            df, window, initial_balance, trading_cost,
            stop_loss=stop_loss,
            take_profit=take_profit,
            min_hold_steps=min_hold_steps,
        )

        # Verify the required indicator columns exist
        missing = [c for c in self.INDICATOR_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"ForexFeatureEnv requires indicator columns missing from df: {missing}\n"
                "Run src/features/indicators.py first."
            )

        # Detect which optional columns are available
        self.has_econ = all(c in self.df.columns for c in self.ECON_COLS)
        self.has_sentiment = all(c in self.df.columns for c in self.SENTIMENT_COLS)
        self.n_extra = (len(self.INDICATOR_COLS)
                        + (len(self.ECON_COLS) if self.has_econ else 0)
                        + (len(self.SENTIMENT_COLS) if self.has_sentiment else 0))

        # Override observation space
        obs_size = window + 2 + self.n_extra
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()            # shape: (window + 2,)

        idx   = self.current_step
        close = float(self.prices[idx]) + 1e-8   # avoid div-by-zero

        rsi       = float(self.df["RSI"].iloc[idx])       / 100.0
        macd      = float(self.df["MACD"].iloc[idx])      / close
        macd_sig  = float(self.df["MACD_signal"].iloc[idx]) / close
        bb_width  = float(self.df["BB_width"].iloc[idx])
        atr       = float(self.df["ATR"].iloc[idx])       / close
        adx       = float(self.df["ADX"].iloc[idx])       / 100.0
        sma_ratio = float(self.df["SMA_ratio"].iloc[idx]) - 1.0   # centre at 0

        extra = [rsi, macd, macd_sig, bb_width, atr, adx, sma_ratio]

        # Append economic features if available
        if self.has_econ:
            rate_diff    = float(self.df["rate_diff"].iloc[idx])
            yield_curve  = float(self.df["yield_curve"].iloc[idx])
            rate_diff_ma = float(self.df["rate_diff_ma"].iloc[idx])
            extra.extend([rate_diff, yield_curve, rate_diff_ma])

        # Append sentiment features if available
        if self.has_sentiment:
            sent_score = float(self.df["sentiment_score"].iloc[idx])
            sent_vol   = float(self.df["sentiment_vol"].iloc[idx])
            news_vol   = float(self.df["news_volume"].iloc[idx])
            extra.extend([sent_score, sent_vol, news_vol])

        extra = np.array(extra, dtype=np.float32)
        return np.concatenate([base_obs, extra])
