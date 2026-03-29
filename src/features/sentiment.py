"""
sentiment.py — Forex news sentiment analysis using RSS feeds + TextBlob.

Scrapes headlines from major forex news sources, scores them for sentiment,
and produces per-pair sentiment features that can be added to the observation.

No API key required — uses public RSS feeds.

Features produced:
  sentiment_score : rolling average sentiment (-1 to +1) for the pair
  sentiment_vol   : volatility of sentiment (disagreement among headlines)
  news_volume     : number of headlines mentioning the pair (normalized)
"""

import re
import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ── RSS feed sources (no API key needed) ───────────────────────────────────

FOREX_RSS_FEEDS = [
    "https://www.forexlive.com/feed",
    "https://www.fxstreet.com/rss",
    "https://www.dailyfx.com/feeds/forex",
    "https://www.investing.com/rss/news_301.rss",
]

# Currency keywords to match headlines to pairs
PAIR_KEYWORDS = {
    "EURUSD": ["eurusd", "eur/usd", "euro", "ecb", "eurozone", "european central bank"],
    "USDJPY": ["usdjpy", "usd/jpy", "yen", "boj", "bank of japan", "japanese"],
    "GBPUSD": ["gbpusd", "gbp/usd", "pound", "sterling", "boe", "bank of england", "british"],
    "AUDUSD": ["audusd", "aud/usd", "aussie", "rba", "reserve bank of australia", "australian"],
    "USDCAD": ["usdcad", "usd/cad", "loonie", "boc", "bank of canada", "canadian", "oil price"],
}

# General USD keywords (affect all pairs)
USD_KEYWORDS = ["dollar", "usd", "fed", "federal reserve", "fomc", "powell",
                "nonfarm", "non-farm", "cpi", "inflation", "treasury"]


def _fetch_rss(url: str, timeout: int = 10) -> list:
    """Fetch and parse an RSS feed, return list of (title, pub_date) tuples."""
    if not HAS_REQUESTS:
        return []
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "ForexBot/1.0 (Student Capstone Project)"
        })
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        items = []
        for item in root.iter("item"):
            title_el = item.find("title")
            date_el = item.find("pubDate")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
                pub_date = date_el.text.strip() if date_el is not None and date_el.text else None
                items.append((title, pub_date))
        return items
    except Exception:
        return []


def _score_headline(headline: str) -> float:
    """
    Score a headline's sentiment using TextBlob.
    Returns a value between -1 (very bearish) and +1 (very bullish).
    """
    if not HAS_TEXTBLOB:
        return 0.0
    blob = TextBlob(headline)
    return blob.sentiment.polarity


def _match_pair(headline: str, pair: str) -> bool:
    """Check if a headline is relevant to a specific currency pair."""
    lower = headline.lower()
    keywords = PAIR_KEYWORDS.get(pair, [])
    return any(kw in lower for kw in keywords)


def _is_usd_related(headline: str) -> bool:
    """Check if headline mentions USD/Fed/macro."""
    lower = headline.lower()
    return any(kw in lower for kw in USD_KEYWORDS)


def fetch_live_sentiment(pairs: list = None) -> dict:
    """
    Fetch live sentiment from RSS feeds.

    Returns
    -------
    dict: {pair: {"sentiment_score": float, "sentiment_vol": float, "news_volume": float}}
    """
    if pairs is None:
        pairs = list(PAIR_KEYWORDS.keys())

    # Collect all headlines
    all_headlines = []
    for url in FOREX_RSS_FEEDS:
        items = _fetch_rss(url)
        all_headlines.extend(items)

    if not all_headlines:
        # Return neutral sentiment if no feeds available
        return {pair: {"sentiment_score": 0.0, "sentiment_vol": 0.0, "news_volume": 0.0}
                for pair in pairs}

    # Score each headline
    scored = []
    for title, pub_date in all_headlines:
        score = _score_headline(title)
        scored.append({"title": title, "score": score, "date": pub_date})

    # Match headlines to pairs and compute features
    result = {}
    for pair in pairs:
        pair_scores = []
        usd_scores = []

        for item in scored:
            if _match_pair(item["title"], pair):
                pair_scores.append(item["score"])
            if _is_usd_related(item["title"]):
                usd_scores.append(item["score"])

        # Combine pair-specific and USD-general sentiment
        all_scores = pair_scores + [s * 0.5 for s in usd_scores]  # USD news weighted 50%

        if all_scores:
            sentiment_score = float(np.mean(all_scores))
            sentiment_vol = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
            news_volume = min(len(pair_scores) / 10.0, 1.0)  # normalize: 10+ articles = 1.0
        else:
            sentiment_score = 0.0
            sentiment_vol = 0.0
            news_volume = 0.0

        # For pairs where USD is the quote currency (EURUSD, GBPUSD, AUDUSD),
        # positive USD sentiment = bearish for the pair
        if pair in ["EURUSD", "GBPUSD", "AUDUSD"] and usd_scores:
            usd_effect = float(np.mean(usd_scores)) * -0.3
            sentiment_score += usd_effect

        # Clip to [-1, 1]
        sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))

        result[pair] = {
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_vol": round(sentiment_vol, 4),
            "news_volume": round(news_volume, 4),
        }

    return result


def generate_historical_sentiment(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Generate synthetic historical sentiment features for training.

    Since we can't get historical RSS data, we approximate sentiment using
    price momentum and volatility as proxies:
      - Positive momentum (5-day return > 0) → mild positive sentiment
      - High volatility → high sentiment disagreement
      - Volume spikes → more news activity

    This is a reasonable proxy because in practice, news sentiment and
    price momentum are correlated (news follows price in the short term).

    Parameters
    ----------
    df : DataFrame with Close, Volume, Return columns
    pair : currency pair name

    Returns
    -------
    DataFrame with added columns: sentiment_score, sentiment_vol, news_volume
    """
    df = df.copy()

    # Momentum proxy: 5-day return, smoothed
    ret_5d = df["Close"].pct_change(5)
    ret_20d = df["Close"].pct_change(20)

    # Sentiment score: blend of short and medium-term momentum
    # Scaled to roughly [-0.5, 0.5] range
    raw_sentiment = (ret_5d * 0.6 + ret_20d * 0.4) * 10
    df["sentiment_score"] = raw_sentiment.rolling(3, min_periods=1).mean().clip(-1, 1)

    # Sentiment volatility: rolling std of daily returns (disagreement proxy)
    df["sentiment_vol"] = df["Return"].rolling(10, min_periods=1).std() * 10
    df["sentiment_vol"] = df["sentiment_vol"].clip(0, 1)

    # News volume: normalized volume spikes
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        vol_ma = df["Volume"].rolling(20, min_periods=1).mean()
        df["news_volume"] = (df["Volume"] / (vol_ma + 1e-8)).clip(0, 2) / 2.0
    else:
        df["news_volume"] = 0.5  # default mid-level

    # Fill any NaN from rolling windows
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    df["sentiment_vol"] = df["sentiment_vol"].fillna(0.0)
    df["news_volume"] = df["news_volume"].fillna(0.5)

    return df


def add_sentiment_to_df(df: pd.DataFrame, pair: str, live: bool = False) -> pd.DataFrame:
    """
    Add sentiment features to a DataFrame.

    Parameters
    ----------
    df   : DataFrame with price data and indicators
    pair : currency pair name
    live : if True, fetch live sentiment from RSS feeds;
           if False, generate historical proxy from price data

    Returns
    -------
    DataFrame with 3 new columns: sentiment_score, sentiment_vol, news_volume
    """
    if live and HAS_TEXTBLOB and HAS_REQUESTS:
        sentiment = fetch_live_sentiment([pair])
        pair_sent = sentiment[pair]

        df = df.copy()
        # For live, apply the current sentiment to the last row
        # and generate historical proxy for prior rows
        df = generate_historical_sentiment(df, pair)
        # Override latest row with live sentiment
        df.loc[df.index[-1], "sentiment_score"] = pair_sent["sentiment_score"]
        df.loc[df.index[-1], "sentiment_vol"] = pair_sent["sentiment_vol"]
        df.loc[df.index[-1], "news_volume"] = pair_sent["news_volume"]
        return df
    else:
        return generate_historical_sentiment(df, pair)
