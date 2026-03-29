"""
oanda_client.py — Wrapper around the OANDA v20 REST API for the practice account.

Handles: fetching candles, placing orders, checking positions, account info.
Uses raw `requests` instead of a third-party OANDA library to keep dependencies light.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OANDA_API_URL = os.getenv("OANDA_API_URL", "https://api-fxpractice.oanda.com")
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")


class OandaClient:
    """
    Lightweight OANDA v20 REST API client for paper trading.

    Usage
    -----
        client = OandaClient()
        candles = client.fetch_candles("EUR_USD", count=250, granularity="D")
        client.place_order("EUR_USD", units=1000)   # buy
        client.place_order("EUR_USD", units=-1000)  # sell
        client.close_position("EUR_USD")
    """

    def __init__(self, api_key=None, account_id=None, api_url=None):
        self.api_key = api_key or OANDA_API_KEY
        self.account_id = account_id or OANDA_ACCOUNT_ID
        self.api_url = api_url or OANDA_API_URL

        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError(
                "OANDA_API_KEY not set. Add it to your .env file.\n"
                "Get one free at https://developer.oanda.com"
            )
        if not self.account_id or self.account_id == "your-account-id-here":
            raise ValueError(
                "OANDA_ACCOUNT_ID not set. Add it to your .env file."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── Account ────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        """Return account balance, unrealised PnL, and open trade count."""
        url = f"{self.api_url}/v3/accounts/{self.account_id}/summary"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        acct = resp.json()["account"]
        return {
            "balance": float(acct["balance"]),
            "unrealized_pnl": float(acct["unrealizedPL"]),
            "open_trades": int(acct["openTradeCount"]),
            "currency": acct["currency"],
        }

    # ── Candles ────────────────────────────────────────────────────────────

    def fetch_candles(
        self,
        instrument: str,
        count: int = 250,
        granularity: str = "D",
    ) -> pd.DataFrame:
        """
        Fetch recent OHLCV candles from OANDA.

        Parameters
        ----------
        instrument  : OANDA format, e.g. "EUR_USD"
        count       : number of candles (max 5000)
        granularity : "D" (daily), "H4", "H1", "M15", etc.

        Returns
        -------
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        url = f"{self.api_url}/v3/instruments/{instrument}/candles"
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M",  # mid prices
        }
        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()

        candles = resp.json()["candles"]
        rows = []
        for c in candles:
            if c["complete"]:  # only use completed candles
                mid = c["mid"]
                rows.append({
                    "Date": pd.Timestamp(c["time"]),
                    "Open": float(mid["o"]),
                    "High": float(mid["h"]),
                    "Low": float(mid["l"]),
                    "Close": float(mid["c"]),
                    "Volume": int(c["volume"]),
                })

        df = pd.DataFrame(rows)
        return df

    # ── Orders ─────────────────────────────────────────────────────────────

    def place_order(self, instrument: str, units: int) -> dict:
        """
        Place a market order.

        Parameters
        ----------
        instrument : e.g. "EUR_USD"
        units      : positive = buy, negative = sell

        Returns
        -------
        dict with order fill details
        """
        url = f"{self.api_url}/v3/accounts/{self.account_id}/orders"
        body = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
            }
        }
        resp = requests.post(url, headers=self.headers, json=body)
        resp.raise_for_status()

        data = resp.json()
        if "orderFillTransaction" in data:
            fill = data["orderFillTransaction"]
            return {
                "instrument": fill["instrument"],
                "units": float(fill["units"]),
                "price": float(fill["price"]),
                "pl": float(fill.get("pl", 0)),
                "time": fill["time"],
            }
        return data

    # ── Positions ──────────────────────────────────────────────────────────

    def get_position(self, instrument: str) -> dict:
        """
        Get the current open position for an instrument.

        Returns
        -------
        dict with 'long_units', 'short_units', 'unrealized_pnl'
        Returns zeros if no position is open.
        """
        url = f"{self.api_url}/v3/accounts/{self.account_id}/positions/{instrument}"
        resp = requests.get(url, headers=self.headers)

        if resp.status_code == 404:
            return {"long_units": 0, "short_units": 0, "unrealized_pnl": 0.0}

        resp.raise_for_status()
        pos = resp.json()["position"]
        return {
            "long_units": float(pos["long"]["units"]),
            "short_units": float(pos["short"]["units"]),
            "unrealized_pnl": float(pos["unrealizedPL"]),
        }

    def close_position(self, instrument: str, side: str = "ALL") -> dict:
        """
        Close an open position.

        Parameters
        ----------
        side : "ALL" closes both long and short, "LONG" or "SHORT" for one side
        """
        url = f"{self.api_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"

        if side == "ALL":
            body = {"longUnits": "ALL", "shortUnits": "ALL"}
        elif side == "LONG":
            body = {"longUnits": "ALL"}
        else:
            body = {"shortUnits": "ALL"}

        resp = requests.put(url, headers=self.headers, json=body)
        resp.raise_for_status()
        return resp.json()

    # ── Convenience ────────────────────────────────────────────────────────

    def get_net_position(self, instrument: str) -> int:
        """Return net position as -1 (short), 0 (flat), or 1 (long)."""
        pos = self.get_position(instrument)
        net = pos["long_units"] + pos["short_units"]
        if net > 0:
            return 1
        elif net < 0:
            return -1
        return 0
