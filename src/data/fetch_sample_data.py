"""
Forex Bot Capstone - Sample Data Fetcher
Run this script to generate sample data for your proposal.

Requirements: pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_sample_data():
    """Fetch sample data for proposal appendix."""

    # 5 Most Traded Forex Pairs (Majors)
    assets = {
        'EURUSD': 'EURUSD=X',   # Euro / US Dollar
        'USDJPY': 'USDJPY=X',   # US Dollar / Japanese Yen
        'GBPUSD': 'GBPUSD=X',   # British Pound / US Dollar
        'AUDUSD': 'AUDUSD=X',   # Australian Dollar / US Dollar
        'USDCAD': 'USDCAD=X',   # US Dollar / Canadian Dollar
    }

    print("=" * 60)
    print("FOREX BOT - SAMPLE DATA FETCHER (5 Major Pairs)")
    print("=" * 60)

    # Fetch EUR/USD sample for proposal (1000 rows of hourly data)
    print("\n📊 Fetching EUR/USD sample data (for proposal appendix)...")

    eurusd = yf.download(
        'EURUSD=X',
        period='60d',
        interval='1h',
        progress=False
    )

    if len(eurusd) > 0:
        eurusd = eurusd.reset_index()

        if isinstance(eurusd.columns, pd.MultiIndex):
            eurusd.columns = [col[0] if col[1] == '' or col[1] == 'EURUSD=X' else col[0] for col in eurusd.columns]

        needed_cols = []
        for col in eurusd.columns:
            col_lower = str(col).lower()
            if 'datetime' in col_lower or 'date' in col_lower:
                needed_cols.append(('Datetime', col))
            elif col_lower == 'open':
                needed_cols.append(('Open', col))
            elif col_lower == 'high':
                needed_cols.append(('High', col))
            elif col_lower == 'low':
                needed_cols.append(('Low', col))
            elif col_lower == 'close' and 'adj' not in col_lower:
                needed_cols.append(('Close', col))
            elif col_lower == 'volume':
                needed_cols.append(('Volume', col))

        clean_data = pd.DataFrame()
        for new_name, old_name in needed_cols:
            clean_data[new_name] = eurusd[old_name]

        if 'Volume' not in clean_data.columns:
            clean_data['Volume'] = 0

        sample = clean_data.head(1000)
        sample.to_csv('sample_data_eurusd.csv', index=False)

        print(f"✅ Saved: sample_data_eurusd.csv ({len(sample)} rows)")
        print(f"   Date range: {sample['Datetime'].iloc[0]} to {sample['Datetime'].iloc[-1]}")
        print(f"\n   Preview:")
        print(sample.head(10).to_string(index=False))
    else:
        print("❌ Failed to fetch EUR/USD data")

    # Fetch all 7 major pairs (daily data, 5 years)
    print("\n" + "=" * 60)
    print("📊 Fetching full dataset (7 Major Forex Pairs, 5 years)...")
    print("=" * 60)

    all_data = {}

    for name, ticker in assets.items():
        print(f"\nFetching {name} ({ticker})...")

        try:
            data = yf.download(
                ticker,
                start='2016-01-01',  # Last 10 years
                end='2026-01-31',
                interval='1d',
                progress=False
            )

            if len(data) > 0:
                data = data.reset_index()

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]

                clean_df = pd.DataFrame()

                for col in data.columns:
                    col_str = str(col).lower()
                    if 'date' in col_str:
                        clean_df['Date'] = data[col]
                    elif col_str == 'open':
                        clean_df['Open'] = data[col]
                    elif col_str == 'high':
                        clean_df['High'] = data[col]
                    elif col_str == 'low':
                        clean_df['Low'] = data[col]
                    elif col_str == 'close' and 'adj' not in col_str:
                        clean_df['Close'] = data[col]
                    elif col_str == 'volume':
                        clean_df['Volume'] = data[col]

                clean_df['Pair'] = name
                all_data[name] = clean_df
                print(f"   ✅ {name}: {len(clean_df)} daily candles")
            else:
                print(f"   ❌ {name}: No data returned")

        except Exception as e:
            print(f"   ❌ {name}: Error - {e}")

    # Combine and save full dataset
    if all_data:
        combined = pd.concat(all_data.values(), ignore_index=True)
        combined.to_csv('forex_dataset_daily.csv', index=False)
        print(f"\n✅ Saved: forex_dataset_daily.csv ({len(combined)} total rows)")

    # Summary statistics for proposal
    print("\n" + "=" * 60)
    print("📈 DATA SUMMARY (for your proposal)")
    print("=" * 60)

    total_rows = 0
    for name, df in all_data.items():
        date_col = df['Date']
        first_date = pd.to_datetime(date_col.iloc[0]).strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
        last_date = pd.to_datetime(date_col.iloc[-1]).strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'
        print(f"\n{name}:")
        print(f"   Rows: {len(df)}")
        print(f"   Date range: {first_date} to {last_date}")
        total_rows += len(df)

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_rows} data points across {len(all_data)} forex pairs")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("✅ DONE! Files created:")
    print("   1. sample_data_eurusd.csv   (for proposal Appendix A)")
    print("   2. forex_dataset_daily.csv  (for project development)")
    print("=" * 60)

if __name__ == "__main__":
    fetch_sample_data()
