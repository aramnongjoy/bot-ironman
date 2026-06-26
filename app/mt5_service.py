import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import MetaTrader5 as mt5

load_dotenv()


class MT5Service:
    def __init__(self, login: int = None, password: str = None, server: str = None, path: str = None):
        self._login = login or int(os.getenv("MT5_LOGIN", 0))
        self._password = password or os.getenv("MT5_PASSWORD", "")
        self._server = server or os.getenv("MT5_SERVER", "")
        self._path = path or os.getenv("MT5_PATH", "")
        self._connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialize MT5 and log in to the trade account."""
        if self._path:
            ok = mt5.initialize(path=self._path)
        else:
            ok = mt5.initialize()

        if not ok:
            raise ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")

        if not mt5.login(login=self._login, password=self._password, server=self._server):
            mt5.shutdown()
            raise PermissionError(f"MT5 login failed: {mt5.last_error()}")

        self._connected = True

    def disconnect(self) -> None:
        """Shut down the MT5 connection."""
        mt5.shutdown()
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() or use the context manager.")

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_info(self, key: str):
        """Return a single field from account_info by name.

        Common keys: 'balance', 'equity', 'margin', 'margin_free',
                     'margin_level', 'profit', 'login', 'name', 'currency'
        """
        self._ensure_connected()
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() returned None: {mt5.last_error()}")
        return info._asdict().get(key)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def send_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float = None,
        sl: float = 0.0,
        tp: float = 0.0,
        deviation: int = 20,
        magic: int = 0,
        comment: str = "",
    ) -> dict:
        """Send a market or pending order.

        Parameters
        ----------
        symbol     : trading symbol, e.g. "EURUSD"
        order_type : "BUY" | "SELL" | "BUY_LIMIT" | "SELL_LIMIT" |
                     "BUY_STOP" | "SELL_STOP"
        volume     : lot size
        price      : required for pending orders; use None for market orders
        sl         : stop-loss price (0 = no SL)
        tp         : take-profit price (0 = no TP)
        deviation  : max allowed slippage in points (market orders)
        magic      : expert-advisor magic number
        comment    : order comment string

        Returns
        -------
        dict with keys: 'retcode', 'order', 'deal', 'comment', 'request'
        """
        self._ensure_connected()

        order_type_map = {
            "BUY":        mt5.ORDER_TYPE_BUY,
            "SELL":       mt5.ORDER_TYPE_SELL,
            "BUY_LIMIT":  mt5.ORDER_TYPE_BUY_LIMIT,
            "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
            "BUY_STOP":   mt5.ORDER_TYPE_BUY_STOP,
            "SELL_STOP":  mt5.ORDER_TYPE_SELL_STOP,
        }

        mt5_type = order_type_map.get(order_type.upper())
        if mt5_type is None:
            raise ValueError(f"Unknown order_type '{order_type}'. "
                             f"Valid values: {list(order_type_map)}")

        is_market = order_type.upper() in ("BUY", "SELL")

        if is_market:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise RuntimeError(f"Cannot get tick for {symbol}: {mt5.last_error()}")
            price = tick.ask if order_type.upper() == "BUY" else tick.bid

        request = {
            "action":    mt5.TRADE_ACTION_DEAL if is_market else mt5.TRADE_ACTION_PENDING,
            "symbol":    symbol,
            "volume":    float(volume),
            "type":      mt5_type,
            "price":     float(price),
            "sl":        float(sl),
            "tp":        float(tp),
            "deviation": int(deviation),
            "magic":     int(magic),
            "comment":   comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send() failed: {mt5.last_error()}")

        result_dict = result._asdict()
        result_dict["request"] = request

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(
                f"Order rejected — retcode {result.retcode}: {result.comment}"
            )

        return result_dict

    # ------------------------------------------------------------------
    # Historical price data
    # ------------------------------------------------------------------

    def copy_rates(
        self,
        symbol: str,
        timeframe: int,
        num_bars: int = 500,
        date_from: datetime = None,
        date_to: datetime = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars and return as a DataFrame.

        Parameters
        ----------
        symbol    : trading symbol, e.g. "EURUSD"
        timeframe : mt5.TIMEFRAME_* constant, e.g. mt5.TIMEFRAME_H1
        num_bars  : number of most-recent bars (used when date_from/date_to are None)
        date_from : start datetime (UTC) — use together with date_to
        date_to   : end datetime (UTC)   — use together with date_from

        Returns
        -------
        DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex (UTC)
        """
        self._ensure_connected()

        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol '{symbol}' not found on this broker.")

        if not info.visible:
            mt5.symbol_select(symbol, True)
            time.sleep(0.5)

        if date_from is not None and date_to is not None:
            rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        else:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)

        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"No data returned for {symbol}: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")[["open", "high", "low", "close", "tick_volume"]]
        df = df.rename(columns={"tick_volume": "volume"})
        return df


# ------------------------------------------------------------------
# Quick demo (run directly: python mt5_service.py)
# ------------------------------------------------------------------

if __name__ == "__main__":
    with MT5Service() as svc:
        balance = svc.get_info("balance")
        equity  = svc.get_info("equity")
        margin  = svc.get_info("margin")
        print(f"Balance : {balance:,.2f}")
        print(f"Equity  : {equity:,.2f}")
        print(f"Margin  : {margin:,.2f}")

        # Example market buy — uncomment to actually send
        # result = svc.send_order("EURUSD", "BUY", volume=0.01)
        # print("Order result:", result)
