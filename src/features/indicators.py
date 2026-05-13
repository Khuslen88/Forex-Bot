"""
indicators.py — Add technical indicators to an OHLCV DataFrame.

Uses the `ta` library (already in requirements.txt).
All indicators operate on a single-pair DataFrame with columns: Open, High, Low, Close.
"""

import pandas as pd
import numpy as np
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a single-pair OHLCV DataFrame with technical indicators.

    Input:  DataFrame with columns Open, High, Low, Close (+ any extras like Date, Pair).
    Output: Copy with added indicator columns; initial NaN rows are dropped.

    Indicators added
    ----------------
    RSI          — momentum oscillator (0-100); >70 = overbought, <30 = oversold
    MACD         — trend-following momentum (MACD line, signal line, histogram)
    BB_*         — Bollinger Bands (upper/middle/lower) + BB_width for volatility squeeze
    ATR          — Average True Range, measures day-to-day volatility in price units
    ADX          — Average Directional Index; >25 = trending, <20 = ranging
    SMA_50/200   — Simple moving averages; crossover is a classic trend signal
    SMA_ratio    — SMA_50 / SMA_200; >1 = uptrend, <1 = downtrend (scale-invariant)
    Return       — Daily percentage change in Close price
    """
    df = df.copy()

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]

    # ── Momentum ─────────────────────────────────────────────────────────────
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # ── Trend / MACD ─────────────────────────────────────────────────────────
    macd_indicator   = ta.trend.MACD(close)
    df["MACD"]       = macd_indicator.macd()
    df["MACD_signal"] = macd_indicator.macd_signal()
    df["MACD_hist"]  = macd_indicator.macd_diff()

    # ── Volatility ────────────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"]  = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"]  = bb.bollinger_lband()
    # BB_width: measures squeeze — low width = low vol = potential breakout ahead
    df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # ── Trend Strength ───────────────────────────────────────────────────────
    # ADX: 0-100; >25 = trending, <20 = ranging — tells model WHEN to trend-follow
    df["ADX"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    # ── Trend / Moving Averages ───────────────────────────────────────────────
    df["SMA_50"]  = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df["SMA_200"] = ta.trend.SMAIndicator(close, window=200).sma_indicator()
    # Scale-invariant ratio: works across EUR/USD (~1.1) and USD/JPY (~140)
    df["SMA_ratio"] = df["SMA_50"] / df["SMA_200"]

    # ── Price Return ─────────────────────────────────────────────────────────
    df["Return"] = close.pct_change()

    # Drop NaN rows from lookback periods (SMA_200 needs ~200 bars)
    df = df.dropna().reset_index(drop=True)

    return df


def load_econ_features(
    econ_path: str,
    forex_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge FRED macroeconomic features into a single-pair forex DataFrame.

    Features added
    --------------
    rate_diff    : ECB rate − Fed Funds rate  (positive = EUR favoured)
    yield_curve  : US 10Y−2Y spread (T10Y2Y); negative = recession signal
    rate_diff_ma : 20-day moving average of rate_diff (smoothed signal)

    Parameters
    ----------
    econ_path : path to fred_data.csv
    forex_df  : single-pair DataFrame already processed by add_indicators()

    Returns
    -------
    DataFrame with three extra columns; rows with no economic data are dropped.
    """
    econ = pd.read_csv(econ_path, index_col=0, parse_dates=True)
    econ.index.name = "Date"
    econ = econ.reset_index()
    econ["Date"] = pd.to_datetime(econ["Date"])

    # Rate differential: ECB minus Fed (EUR-USD fundamental driver)
    econ["rate_diff"] = econ["ECBMRRFR"].ffill() - econ["FEDFUNDS"].ffill()

    # Yield curve: 10Y - 2Y spread (recession / risk-on indicator)
    econ["yield_curve"] = econ["T10Y2Y"].ffill()

    econ_slim = econ[["Date", "rate_diff", "yield_curve"]].dropna().sort_values("Date")

    # merge_asof with backward direction: each forex bar gets the most recent
    # FRED reading on/before its timestamp. Works for both daily and hourly
    # forex data (hourly bars carry forward the prior day's macro values).
    forex_sorted = forex_df.sort_values("Date").reset_index(drop=True)

    # Normalize timezones — Dukascopy returns tz-aware UTC, FRED is tz-naive.
    # merge_asof requires matching dtypes, so drop tz info from forex side.
    if pd.api.types.is_datetime64tz_dtype(forex_sorted["Date"]):
        forex_sorted["Date"] = forex_sorted["Date"].dt.tz_localize(None)
    merged = pd.merge_asof(
        forex_sorted, econ_slim, on="Date", direction="backward",
    )
    merged = merged.dropna(subset=["rate_diff", "yield_curve"])

    # Smoothed rate differential — window adapts to data length so it stays meaningful
    # for both daily (~20 days) and hourly (~480 hours = 20 days) timeframes.
    merged["rate_diff_ma"] = merged["rate_diff"].rolling(20, min_periods=1).mean()

    return merged.reset_index(drop=True)


def load_pair(
    df_all: pd.DataFrame,
    pair: str,
    econ_path: str = None,
    sentiment: bool = True,
    denoise: str = None,
    denoise_kwargs: dict = None,
) -> pd.DataFrame:
    """
    Filter a single currency pair from the combined forex DataFrame and add indicators.

    Parameters
    ----------
    df_all          : combined DataFrame (from forex_dataset_daily.csv) with a 'Pair' column
    pair            : e.g. 'EURUSD', 'USDJPY'
    econ_path       : optional path to fred_data.csv; if provided, merges macro features
    sentiment       : if True, add historical sentiment proxy features
    denoise         : None | "wavelet" | "emd" — DSP-based denoising of OHLC before indicators
    denoise_kwargs  : dict forwarded to denoise_ohlc_causal (window, refresh_every, ...)

    When `denoise` is set, OHLC columns are filtered with a causal rolling-window
    wavelet or EMD denoiser BEFORE technical indicators are computed. Method
    ported from the Modern Topics in DS DSP-RL project (May 2026).
    """
    df = df_all[df_all["Pair"] == pair].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Optional DSP denoising (causal — no look-ahead bias) ─────────────
    if denoise and denoise != "none":
        from src.features.denoising import denoise_ohlc_causal
        kwargs = {"window": 256, "refresh_every": 1}
        if denoise == "emd":
            # EMD is ~50x slower than wavelet; only refresh once per day on 1H
            kwargs["refresh_every"] = 24
            kwargs["n_imfs_to_remove"] = 1
        if denoise_kwargs:
            kwargs.update(denoise_kwargs)
        df = denoise_ohlc_causal(df, method=denoise, **kwargs)

    df = add_indicators(df)
    if econ_path:
        df = load_econ_features(econ_path, df)
    if sentiment:
        from src.features.sentiment import add_sentiment_to_df
        df = add_sentiment_to_df(df, pair, live=False)
    return df
