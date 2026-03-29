"""
paper_client.py — Paper trading client that simulates a demo account.

Uses Yahoo Finance for real market prices. Tracks positions and balance
in a local JSON file so state persists between runs.

Same interface as OandaClient / MT5Client, so live_trader.py works with
any backend.
"""

import os
import json
import datetime
import pandas as pd
import yfinance as yf

STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "paper_account.json",
)

# Yahoo Finance symbols for forex pairs
YAHOO_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
}

DEFAULT_STATE = {
    "balance": 100_000.0,
    "starting_balance": 100_000.0,
    "positions": {},       # symbol -> {"units": int, "entry_price": float}
    "trade_history": [],   # list of completed trades
    "created": None,
}


class PaperClient:
    """
    Simulated broker for paper trading on Mac.

    Usage
    -----
        client = PaperClient()
        candles = client.fetch_candles("EURUSD", count=250, granularity="D")
        client.place_order("EURUSD", units=1000)   # buy
        client.place_order("EURUSD", units=-1000)  # sell
        client.close_position("EURUSD")
    """

    def __init__(self, state_file: str = None):
        self.state_file = os.path.normpath(state_file or STATE_FILE)
        self.state = self._load_state()

    # ── State persistence ──────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        state = DEFAULT_STATE.copy()
        state["created"] = datetime.datetime.now().isoformat()
        self._save_state(state)
        return state

    def _save_state(self, state: dict = None):
        state = state or self.state
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    # ── Account ────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        unrealized = self._calc_unrealized_pnl()
        return {
            "balance": self.state["balance"],
            "unrealized_pnl": unrealized,
            "equity": self.state["balance"] + unrealized,
            "open_trades": len(self.state["positions"]),
            "currency": "USD",
            "total_trades": len(self.state["trade_history"]),
        }

    def _calc_unrealized_pnl(self) -> float:
        total = 0.0
        for symbol, pos in self.state["positions"].items():
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue
            units = pos["units"]
            entry = pos["entry_price"]
            if units > 0:
                total += (current_price - entry) * abs(units)
            else:
                total += (entry - current_price) * abs(units)
        return round(total, 2)

    def _get_current_price(self, symbol: str) -> float:
        yahoo_sym = YAHOO_SYMBOLS.get(symbol, f"{symbol}=X")
        try:
            ticker = yf.Ticker(yahoo_sym)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return None

    # ── Candles ────────────────────────────────────────────────────────────

    def fetch_candles(
        self,
        symbol: str,
        count: int = 250,
        granularity: str = "D",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Yahoo Finance.

        Parameters
        ----------
        symbol      : e.g. "EURUSD"
        count       : approximate number of candles
        granularity : "D" (daily), "H1" (1 hour), "H4" (4 hour)
        """
        yahoo_sym = YAHOO_SYMBOLS.get(symbol, f"{symbol}=X")

        # Map granularity to yfinance parameters
        interval_map = {"D": "1d", "H1": "1h", "H4": "1h", "M15": "15m"}
        interval = interval_map.get(granularity, "1d")

        # Estimate period needed
        if interval == "1d":
            days = int(count * 1.5)  # buffer for weekends/holidays
            period = f"{days}d" if days <= 730 else "2y"
        elif interval == "1h":
            period = "60d"  # yfinance max for hourly
        else:
            period = "30d"

        ticker = yf.Ticker(yahoo_sym)
        data = ticker.history(period=period, interval=interval)

        if data.empty:
            raise ValueError(f"No data returned for {yahoo_sym}")

        df = data.reset_index()

        # Normalize column names
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        # Keep only OHLCV columns
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

        # Trim to requested count
        if len(df) > count:
            df = df.tail(count).reset_index(drop=True)

        return df

    # ── Orders ─────────────────────────────────────────────────────────────

    def place_order(self, symbol: str, units: int) -> dict:
        """
        Simulate a market order fill at the current price.

        Parameters
        ----------
        symbol : e.g. "EURUSD"
        units  : positive = buy, negative = sell
        """
        price = self._get_current_price(symbol)
        if price is None:
            raise RuntimeError(f"Cannot get current price for {symbol}")

        # If there's an existing position in the opposite direction, close it first
        existing = self.state["positions"].get(symbol)
        realized_pl = 0.0

        if existing:
            old_units = existing["units"]
            entry = existing["entry_price"]

            # Calculate P&L from closing the old position
            if old_units > 0:
                realized_pl = (price - entry) * abs(old_units)
            else:
                realized_pl = (entry - price) * abs(old_units)

            self.state["balance"] += realized_pl
            self.state["trade_history"].append({
                "symbol": symbol,
                "units": old_units,
                "entry_price": entry,
                "exit_price": price,
                "pl": round(realized_pl, 2),
                "closed_at": datetime.datetime.now().isoformat(),
            })
            del self.state["positions"][symbol]

        # Open new position
        self.state["positions"][symbol] = {
            "units": units,
            "entry_price": price,
            "opened_at": datetime.datetime.now().isoformat(),
        }

        self._save_state()

        return {
            "instrument": symbol,
            "units": float(units),
            "price": price,
            "pl": round(realized_pl, 2),
            "time": datetime.datetime.now().isoformat(),
        }

    # ── Positions ──────────────────────────────────────────────────────────

    def get_position(self, symbol: str) -> dict:
        pos = self.state["positions"].get(symbol)
        if pos is None:
            return {"long_units": 0, "short_units": 0, "unrealized_pnl": 0.0}

        units = pos["units"]
        current = self._get_current_price(symbol)
        pnl = 0.0
        if current:
            if units > 0:
                pnl = (current - pos["entry_price"]) * abs(units)
            else:
                pnl = (pos["entry_price"] - current) * abs(units)

        return {
            "long_units": units if units > 0 else 0,
            "short_units": units if units < 0 else 0,
            "unrealized_pnl": round(pnl, 2),
        }

    def close_position(self, symbol: str, side: str = "ALL") -> dict:
        pos = self.state["positions"].get(symbol)
        if pos is None:
            return {"message": "No position to close"}

        price = self._get_current_price(symbol)
        units = pos["units"]
        entry = pos["entry_price"]

        if units > 0:
            pl = (price - entry) * abs(units)
        else:
            pl = (entry - price) * abs(units)

        self.state["balance"] += pl
        self.state["trade_history"].append({
            "symbol": symbol,
            "units": units,
            "entry_price": entry,
            "exit_price": price,
            "pl": round(pl, 2),
            "closed_at": datetime.datetime.now().isoformat(),
        })
        del self.state["positions"][symbol]
        self._save_state()

        return {"symbol": symbol, "pl": round(pl, 2), "price": price}

    def get_net_position(self, symbol: str) -> int:
        pos = self.state["positions"].get(symbol)
        if pos is None:
            return 0
        return 1 if pos["units"] > 0 else -1

    # ── Display ────────────────────────────────────────────────────────────

    def print_status(self):
        """Print a summary of the paper account."""
        acct = self.get_account_summary()
        pnl = acct["balance"] - self.state["starting_balance"]
        pnl_pct = (pnl / self.state["starting_balance"]) * 100

        print(f"\n  {'─' * 45}")
        print(f"  Paper Account Status")
        print(f"  {'─' * 45}")
        print(f"  Balance:       ${acct['balance']:>12,.2f}")
        print(f"  Unrealized PL: ${acct['unrealized_pnl']:>12,.2f}")
        print(f"  Equity:        ${acct['equity']:>12,.2f}")
        print(f"  Total P&L:     ${pnl:>12,.2f}  ({pnl_pct:+.2f}%)")
        print(f"  Open trades:   {acct['open_trades']}")
        print(f"  Total trades:  {acct['total_trades']}")
        print(f"  {'─' * 45}")

    def reset_account(self, balance: float = 100_000.0):
        """Reset the paper account to starting state."""
        self.state = DEFAULT_STATE.copy()
        self.state["balance"] = balance
        self.state["starting_balance"] = balance
        self.state["created"] = datetime.datetime.now().isoformat()
        self._save_state()
        print(f"  Account reset to ${balance:,.2f}")
