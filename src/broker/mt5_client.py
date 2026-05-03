"""
mt5_client.py — Wrapper around MetaTrader 5 Python API for demo account trading.

Requirements:
  pip install MetaTrader5
  NOTE: Only works on Windows (or Windows VM / Parallels on Mac)
  MT5 terminal must be running and logged into a demo account from any MT5 broker.

Usage:
    client = MT5Client()
    candles = client.fetch_candles("EURUSD", count=250)
    client.place_order("EURUSD", units=1000)    # buy
    client.place_order("EURUSD", units=-1000)   # sell
    client.close_position("EURUSD")
"""

import datetime
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


# MT5 timeframe mapping
TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1  if mt5 else 1,
    "M5":  mt5.TIMEFRAME_M5  if mt5 else 5,
    "M15": mt5.TIMEFRAME_M15 if mt5 else 15,
    "H1":  mt5.TIMEFRAME_H1  if mt5 else 16385,
    "H4":  mt5.TIMEFRAME_H4  if mt5 else 16388,
    "D":   mt5.TIMEFRAME_D1  if mt5 else 16408,
}


class MT5Client:
    """
    Lightweight MetaTrader 5 client for paper trading.

    The MT5 terminal must be running and logged into your broker account
    before calling any methods.
    """

    def __init__(self):
        if mt5 is None:
            raise ImportError(
                "MetaTrader5 package not installed. Run:\n"
                "  pip install MetaTrader5\n"
                "NOTE: Only works on Windows."
            )

        if not mt5.initialize():
            error = mt5.last_error()
            raise ConnectionError(
                f"MT5 initialize() failed: {error}\n"
                "Make sure MetaTrader 5 terminal is running and logged in."
            )

        info = mt5.account_info()
        if info is None:
            raise ConnectionError("Could not get account info. Is MT5 logged in?")

        print(f"  Connected to MT5: {info.server}")
        print(f"  Account: {info.login} ({info.name})")
        print(f"  Balance: ${info.balance:,.2f} ({info.currency})")
        self._account_info = info

    # ── Account ────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        """Return account balance, unrealised PnL, and open trade count."""
        info = mt5.account_info()
        positions = mt5.positions_total()
        return {
            "balance": info.balance,
            "unrealized_pnl": info.profit,
            "open_trades": positions if positions else 0,
            "currency": info.currency,
        }

    # ── Candles ────────────────────────────────────────────────────────────

    def fetch_candles(
        self,
        symbol: str,
        count: int = 250,
        granularity: str = "D",
    ) -> pd.DataFrame:
        """
        Fetch recent OHLCV candles from MT5.

        Parameters
        ----------
        symbol      : MT5 symbol, e.g. "EURUSD" (check your broker's naming)
        count       : number of candles
        granularity : "D", "H4", "H1", "M15", "M5", "M1"

        Returns
        -------
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        tf = TIMEFRAMES.get(granularity)
        if tf is None:
            raise ValueError(f"Unknown granularity '{granularity}'. Use: {list(TIMEFRAMES.keys())}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise ValueError(
                f"No candles returned for {symbol}. Error: {error}\n"
                "Check that the symbol name matches your broker (e.g. 'EURUSD' vs 'EUR/USD')."
            )

        df = pd.DataFrame(rates)
        df["Date"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume",
        })
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return df

    # ── Orders ─────────────────────────────────────────────────────────────

    def place_order(self, symbol: str, units: int) -> dict:
        """
        Place a market order.

        Parameters
        ----------
        symbol : e.g. "EURUSD"
        units  : positive = buy, negative = sell

        Returns
        -------
        dict with order fill details
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol '{symbol}' not found. Check broker symbol names.")

        if not info.visible:
            mt5.symbol_select(symbol, True)

        order_type = mt5.ORDER_TYPE_BUY if units > 0 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol)
        fill_price = price.ask if units > 0 else price.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": abs(units) / 100000.0,  # convert units to lots (1 lot = 100,000)
            "type": order_type,
            "price": fill_price,
            "deviation": 20,  # max slippage in points
            "magic": 123456,  # EA identifier
            "comment": "forex-rl-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(
                f"Order failed: {result.retcode} — {result.comment}"
            )

        return {
            "instrument": symbol,
            "units": units,
            "price": result.price,
            "pl": 0.0,
            "time": datetime.datetime.now().isoformat(),
            "order_id": result.order,
        }

    # ── Positions ──────────────────────────────────────────────────────────

    def get_position(self, symbol: str) -> dict:
        """
        Get the current open position for a symbol.

        Returns
        -------
        dict with 'long_units', 'short_units', 'unrealized_pnl'
        """
        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:
            return {"long_units": 0, "short_units": 0, "unrealized_pnl": 0.0}

        long_units = 0.0
        short_units = 0.0
        total_pnl = 0.0

        for pos in positions:
            volume_units = pos.volume * 100000  # lots back to units
            if pos.type == mt5.ORDER_TYPE_BUY:
                long_units += volume_units
            else:
                short_units -= volume_units
            total_pnl += pos.profit

        return {
            "long_units": long_units,
            "short_units": short_units,
            "unrealized_pnl": total_pnl,
        }

    def close_position(self, symbol: str, side: str = "ALL") -> dict:
        """Close open positions for a symbol."""
        positions = mt5.positions_get(symbol=symbol)

        if positions is None or len(positions) == 0:
            return {"message": "No positions to close"}

        results = []
        for pos in positions:
            if side == "LONG" and pos.type != mt5.ORDER_TYPE_BUY:
                continue
            if side == "SHORT" and pos.type != mt5.ORDER_TYPE_SELL:
                continue

            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol)
            close_price = price.bid if pos.type == mt5.ORDER_TYPE_BUY else price.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 123456,
                "comment": "forex-rl-bot close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            results.append({
                "ticket": pos.ticket,
                "retcode": result.retcode,
                "comment": result.comment,
            })

        return results

    # ── Convenience ────────────────────────────────────────────────────────

    def get_net_position(self, symbol: str) -> int:
        """Return net position as -1 (short), 0 (flat), or 1 (long)."""
        pos = self.get_position(symbol)
        net = pos["long_units"] + pos["short_units"]
        if net > 0:
            return 1
        elif net < 0:
            return -1
        return 0

    def shutdown(self):
        """Disconnect from MT5."""
        mt5.shutdown()
