"""
Live trading loop — EURUSDm H4, Long + Short
Best params from backtest optimization (backtest.py).

Run:  python live.py
Stop: Ctrl+C
"""

import time
import logging
import MetaTrader5 as mt5
import pandas as pd

from app import MT5Service
from app.signals import latest_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
SYMBOL       = "EURUSDm"
TIMEFRAME    = mt5.TIMEFRAME_H4
LOOKBACK     = 200          # bars ที่ดึงมาคำนวณ indicator
VOLUME       = 0.01         # lot size
SL_PCT       = 0.01         # 1% stop loss  (ตรงกับ backtest sl_stop=0.01)
TP_PCT       = 0.02         # 2% take profit (ตรงกับ backtest tp_stop=0.02)
MAGIC        = 20240101     # unique ID — ใช้กรองว่า position ไหนเปิดโดย bot นี้
POLL_SECONDS = 5 * 60       # poll ทุก 5 นาที (H4 = 240 นาที)

# params ที่ได้จาก optimize() แยก long/short
LONG_PARAMS = {
    "rsi_period": 11,
    "ema_fast":    7,
}
SHORT_PARAMS = {
    "rsi_period_short": 11,
    "ema_fast_short":   14,
}
# ────────────────────────────────────────────────────────────────────────────


def fetch_df() -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    return df.rename(columns={"tick_volume": "volume"})


def get_position() -> str | None:
    """คืน 'BUY', 'SELL' หรือ None สำหรับ position ที่ bot เปิดไว้."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return None
    for pos in positions:
        if pos.magic == MAGIC:
            return "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    return None


def close_position() -> None:
    """ปิด position ทั้งหมดของ symbol นี้ที่ magic ตรงกัน."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return
    for pos in positions:
        if pos.magic != MAGIC:
            continue
        tick = mt5.symbol_info_tick(SYMBOL)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       SYMBOL,
            "volume":       pos.volume,
            "type":         close_type,
            "position":     pos.ticket,
            "price":        price,
            "deviation":    20,
            "magic":        MAGIC,
            "comment":      "close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info("Closed position #%d  %s @ %.5f", pos.ticket,
                     "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL", price)
        else:
            log.error("Close failed: %s", result.comment)


def open_order(svc: MT5Service, direction: str) -> None:
    """เปิด order BUY หรือ SELL พร้อม SL/TP แบบ percentage."""
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if direction == "BUY" else tick.bid

    if direction == "BUY":
        sl = round(price * (1 - SL_PCT), 5)
        tp = round(price * (1 + TP_PCT), 5)
    else:
        sl = round(price * (1 + SL_PCT), 5)
        tp = round(price * (1 - TP_PCT), 5)

    svc.send_order(
        symbol=SYMBOL,
        order_type=direction,
        volume=VOLUME,
        sl=sl,
        tp=tp,
        magic=MAGIC,
        comment=f"bot-{direction.lower()}",
    )
    log.info("Opened %-4s @ %.5f  SL=%.5f  TP=%.5f", direction, price, sl, tp)


def run_live():
    log.info("Starting live loop — %s H4  vol=%.2f", SYMBOL, VOLUME)
    last_bar_time = None

    with MT5Service() as svc:
        balance = svc.get_info("balance")
        log.info("Account balance: %.2f %s", balance, svc.get_info("currency"))

        while True:
            try:
                df = fetch_df()
                bar_time = df.index[-2]     # bar ปิดล่าสุด (ไม่ใช่ bar ที่กำลังก่อตัว)

                if bar_time == last_bar_time:
                    time.sleep(POLL_SECONDS)
                    continue

                last_bar_time = bar_time
                log.info("New bar: %s  close=%.5f", bar_time, df["close"].iloc[-2])

                sig = latest_signal(df, **LONG_PARAMS, **SHORT_PARAMS)
                pos = get_position()

                log.info("Signal: %s  |  Position: %s", sig, pos)

                # ── ฝั่ง Long ──────────────────────────────────────────────
                if sig["long_entry"] and pos != "BUY":
                    if pos == "SELL":
                        close_position()
                    open_order(svc, "BUY")

                elif sig["long_exit"] and pos == "BUY":
                    close_position()

                # ── ฝั่ง Short ─────────────────────────────────────────────
                if sig["short_entry"] and pos != "SELL":
                    if pos == "BUY":
                        close_position()
                    open_order(svc, "SELL")

                elif sig["short_exit"] and pos == "SELL":
                    close_position()

            except KeyboardInterrupt:
                log.info("Stopped by user.")
                break
            except Exception as exc:
                log.error("Error: %s", exc, exc_info=True)
                time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_live()
