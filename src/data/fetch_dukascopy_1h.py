"""
fetch_dukascopy_1h.py — Download 10 years of 1H OHLCV forex data from Dukascopy.

Outputs a single CSV at data/raw/forex_dataset_1h.csv with the same schema as
forex_dataset_daily.csv so it can be loaded by load_pair() unchanged:
    Date, Close, High, Low, Open, Volume, Pair

Dukascopy serves data for free with no API key — it streams from their public
historical data feed. Fetches in 1-year chunks per pair to stay within their
default 30k-row request limit (~6,000 1H candles per year per pair).

Usage:
    python src/data/fetch_dukascopy_1h.py
    python src/data/fetch_dukascopy_1h.py --years 5         # last 5 years only
    python src/data/fetch_dukascopy_1h.py --pairs EURUSD    # single pair
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import pandas as pd
import dukascopy_python as dp
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_USD_JPY,
    INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_AUD_USD,
    INSTRUMENT_FX_MAJORS_USD_CAD,
)

PAIR_MAP = {
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
    "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
    "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
    "AUDUSD": INSTRUMENT_FX_MAJORS_AUD_USD,
    "USDCAD": INSTRUMENT_FX_MAJORS_USD_CAD,
}


def fetch_pair_chunked(pair: str, years: int) -> pd.DataFrame:
    """Fetch 1H candles for one pair in 1-year chunks."""
    instrument = PAIR_MAP[pair]
    end = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365 * years)

    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=365), end)
        print(f"  [{pair}] {chunk_start.date()} → {chunk_end.date()}")
        df = dp.fetch(
            instrument,
            dp.INTERVAL_HOUR_1,
            dp.OFFER_SIDE_BID,
            chunk_start,
            chunk_end,
        )
        if df is not None and len(df) > 0:
            chunks.append(df)
        chunk_start = chunk_end

    if not chunks:
        return pd.DataFrame()

    full = pd.concat(chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full


def normalize_to_project_schema(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Convert Dukascopy output to the project's CSV schema."""
    out = df.reset_index().rename(columns={
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    out["Pair"] = pair
    out = out[["Date", "Close", "High", "Low", "Open", "Volume", "Pair"]]
    return out


def main():
    parser = argparse.ArgumentParser(description="Download 1H forex data from Dukascopy")
    parser.add_argument("--years", type=int, default=10,
                        help="Years of history to download (default: 10)")
    parser.add_argument("--pairs", nargs="+", default=list(PAIR_MAP.keys()),
                        help="Pairs to fetch (default: all 5)")
    parser.add_argument("--out", default="data/raw/forex_dataset_1h.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    invalid = [p for p in args.pairs if p not in PAIR_MAP]
    if invalid:
        print(f"ERROR: Unknown pairs: {invalid}. Supported: {list(PAIR_MAP.keys())}")
        sys.exit(1)

    print(f"Fetching {args.years} years of 1H data for {len(args.pairs)} pairs...")
    print(f"Pairs: {args.pairs}")
    print(f"Output: {args.out}\n")

    all_dfs = []
    for pair in args.pairs:
        print(f"--- {pair} ---")
        raw = fetch_pair_chunked(pair, args.years)
        if raw.empty:
            print(f"  WARNING: no data returned for {pair}")
            continue
        normed = normalize_to_project_schema(raw, pair)
        print(f"  rows: {len(normed):,}  range: {normed['Date'].min()} → {normed['Date'].max()}\n")
        all_dfs.append(normed)

    if not all_dfs:
        print("ERROR: no data fetched.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"Done. Wrote {len(combined):,} rows to {args.out}")


if __name__ == "__main__":
    main()
