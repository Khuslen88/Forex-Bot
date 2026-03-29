"""
Fetch Economic Data from FRED (Federal Reserve Economic Data)

Usage:
    python fetch_economic_data.py --api-key YOUR_API_KEY

Or set environment variable:
    export FRED_API_KEY=your_key_here
    python fetch_economic_data.py
"""

import os
import argparse
import pandas as pd
from datetime import datetime
from fredapi import Fred

# =============================================================================
# ECONOMIC INDICATORS FOR FOREX TRADING
# =============================================================================

INDICATORS = {
    # US Indicators (affects USD pairs)
    'FEDFUNDS': {
        'name': 'Federal Funds Rate',
        'description': 'US interest rate - major USD driver',
        'frequency': 'monthly'
    },
    'CPIAUCSL': {
        'name': 'Consumer Price Index (CPI)',
        'description': 'US inflation - affects Fed policy expectations',
        'frequency': 'monthly'
    },
    'UNRATE': {
        'name': 'Unemployment Rate',
        'description': 'US labor market strength',
        'frequency': 'monthly'
    },
    'GDP': {
        'name': 'Gross Domestic Product',
        'description': 'US economic growth',
        'frequency': 'quarterly'
    },
    'T10Y2Y': {
        'name': '10Y-2Y Treasury Spread',
        'description': 'Yield curve - recession indicator',
        'frequency': 'daily'
    },
    'DTWEXBGS': {
        'name': 'Trade Weighted US Dollar Index',
        'description': 'Overall USD strength',
        'frequency': 'daily'
    },

    # For other currencies
    'IR3TIB01JPM156N': {
        'name': 'Japan 3-Month Interbank Rate',
        'description': 'Japanese interest rate proxy',
        'frequency': 'monthly'
    },
    'ECBMRRFR': {
        'name': 'ECB Main Refinancing Rate',
        'description': 'Euro area interest rate',
        'frequency': 'monthly'
    },
    'BOGZ1FL073164003Q': {
        'name': 'Bank of England Rate',
        'description': 'UK interest rate',
        'frequency': 'quarterly'
    },
}


def fetch_fred_data(api_key: str, start_date: str = '2016-01-01', end_date: str = None):
    """
    Fetch all economic indicators from FRED.

    Args:
        api_key: FRED API key
        start_date: Start date for data
        end_date: End date (defaults to today)

    Returns:
        DataFrame with all indicators
    """

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    fred = Fred(api_key=api_key)

    print("=" * 60)
    print("FETCHING FRED ECONOMIC DATA")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print()

    all_data = {}

    for series_id, info in INDICATORS.items():
        print(f"📊 Fetching {info['name']}...")

        try:
            data = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )

            if len(data) > 0:
                all_data[series_id] = data
                print(f"   ✅ {len(data)} observations")
            else:
                print(f"   ⚠️ No data returned")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Combine into DataFrame
    if all_data:
        df = pd.DataFrame(all_data)

        # Forward fill missing values (for different frequencies)
        df = df.ffill()

        print()
        print("=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print()
        print(df.tail())

        return df

    return None


def save_economic_data(df: pd.DataFrame, output_path: str):
    """Save economic data to CSV."""
    df.to_csv(output_path)
    print(f"\n✅ Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fetch FRED economic data')
    parser.add_argument('--api-key', type=str, help='FRED API key')
    parser.add_argument('--start', type=str, default='2016-01-01', help='Start date')
    parser.add_argument('--end', type=str, default=None, help='End date')
    parser.add_argument('--output', type=str, default=None, help='Output file path')

    args = parser.parse_args()

    # Get API key from argument or environment
    api_key = args.api_key or os.environ.get('FRED_API_KEY')

    if not api_key:
        print("❌ Error: FRED API key required!")
        print()
        print("Get your free API key at:")
        print("https://fred.stlouisfed.org/docs/api/api_key.html")
        print()
        print("Then run:")
        print("  python fetch_economic_data.py --api-key YOUR_KEY")
        print()
        print("Or set environment variable:")
        print("  export FRED_API_KEY=your_key_here")
        return

    # Fetch data
    df = fetch_fred_data(api_key, args.start, args.end)

    if df is not None:
        # Default output path
        output_path = args.output
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            output_path = os.path.join(project_root, 'data', 'economic', 'fred_data.csv')

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save
        save_economic_data(df, output_path)

        print()
        print("=" * 60)
        print("✅ DONE!")
        print("=" * 60)


if __name__ == "__main__":
    main()
