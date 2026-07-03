"""
Live trading loop — symbol/timeframe/params loaded from config.json

Run:  python live.py
Stop: Ctrl+C
"""

import json
import time
import logging
import MetaTrader5 as mt5
import pandas as pd

from app import MT5Service
from app.signals import latest_signal
from app.line_notify import send as line_send

CONFIG_FILE = "config.json"


TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def load_run_config() -> dict:
    """Load symbol, timeframe, SL/TP, and indicator params from config.json."""
    cfg = load_config()
    tf_str = cfg.get("timeframe", "H4")
    tf = TF_MAP.get(tf_str)
    if tf is None:
        raise ValueError(f"Unknown timeframe '{tf_str}' in config.json. Valid: {list(TF_MAP)}")

    lp = cfg.get("long_params", {})
    sp = cfg.get("short_params", {})

    return {
        "symbol":    cfg.get("symbol", "EURUSDc"),
        "timeframe": tf,
        "tf_str":    tf_str,
        "sl_pct":    cfg.get("sl_pct", 0.01),
        "tp_pct":    cfg.get("tp_pct", 0.02),
        "long_params": {
            k: lp[k] for k in
            ["rsi_period", "ema_fast", "ema_slow", "ema_trend",
             "rsi_overbought", "rsi_momentum"]
            if k in lp
        },
        "short_params": {
            k: sp[k] for k in ["rsi_period_short", "ema_fast_short"] if k in sp
        },
    }


def is_enabled() -> bool:
    return load_config().get("is_enable", True)


def calc_volume(equity: float, base_equity: float) -> float:
    """คำนวณ lot size จาก config.

    dynamic : scale ตาม equity ปัจจุบัน เทียบกับ base_equity
    fixed   : ใช้ lot_size คงที่
    """
    cfg = load_config()
    base_lot = cfg.get("lot_size", 0.01)
    if cfg.get("lot_size_type") == "dynamic":
        lot = base_lot * (equity / base_equity)
        lot = max(0.01, round(lot / 0.01) * 0.01)
        return lot
    return base_lot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LOOKBACK     = 200
MAGIC        = 20240101
POLL_SECONDS = 5 * 60


def fetch_df(symbol: str, timeframe: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    return df.rename(columns={"tick_volume": "volume"})


def get_position(symbol: str) -> str | None:
    """คืน 'BUY', 'SELL' หรือ None สำหรับ position ที่ bot เปิดไว้."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for pos in positions:
        if pos.magic == MAGIC:
            return "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    return None


def close_position(symbol: str) -> None:
    """ปิด position ทั้งหมดของ symbol นี้ที่ magic ตรงกัน."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for pos in positions:
        if pos.magic != MAGIC:
            continue
        tick = mt5.symbol_info_tick(symbol)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
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
            side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
            log.info("Closed position #%d  %s @ %.5f", pos.ticket, side, price)
            line_send(f"⬜ CLOSE {side}  {symbol}\nPrice : {price:.5f}")
        else:
            log.error("Close failed: %s", result.comment)


def open_order(svc: MT5Service, symbol: str, direction: str,
               sl_pct: float, tp_pct: float,
               equity: float, base_equity: float) -> None:
    """เปิด order BUY หรือ SELL พร้อม SL/TP แบบ percentage."""
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if direction == "BUY" else tick.bid
    volume = calc_volume(equity, base_equity)

    if direction == "BUY":
        sl = round(price * (1 - sl_pct), 5)
        tp = round(price * (1 + tp_pct), 5)
    else:
        sl = round(price * (1 + sl_pct), 5)
        tp = round(price * (1 - tp_pct), 5)

    svc.send_order(
        symbol=symbol,
        order_type=direction,
        volume=volume,
        sl=sl,
        tp=tp,
        magic=MAGIC,
        comment=f"bot-{direction.lower()}",
    )
    log.info("Opened %-4s @ %.5f  lot=%.2f  SL=%.5f  TP=%.5f", direction, price, volume, sl, tp)
    line_send(
        f"{'🟢 BUY' if direction == 'BUY' else '🔴 SELL'}  {symbol}\n"
        f"Price : {price:.5f}\n"
        f"Lot   : {volume:.2f}\n"
        f"SL    : {sl:.5f}\n"
        f"TP    : {tp:.5f}"
    )


def run_live():
    rc = load_run_config()
    symbol      = rc["symbol"]
    timeframe   = rc["timeframe"]
    tf_str      = rc["tf_str"]
    sl_pct      = rc["sl_pct"]
    tp_pct      = rc["tp_pct"]
    long_params  = rc["long_params"]
    short_params = rc["short_params"]

    log.info("Starting live loop — %s %s", symbol, tf_str)
    log.info("Long params:  %s", long_params)
    log.info("Short params: %s", short_params)
    line_send(
        f"🤖 Bot started\n"
        f"Symbol    : {symbol}\n"
        f"Timeframe : {tf_str}\n"
        f"Long RSI={long_params.get('rsi_period','—')} EMAf={long_params.get('ema_fast','—')}\n"
        f"Short RSI={short_params.get('rsi_period_short','—')} EMAf={short_params.get('ema_fast_short','—')}"
    )

    last_bar_time = None

    with MT5Service() as svc:
        balance = svc.get_info("balance")
        log.info("Account balance: %.2f %s", balance, svc.get_info("currency"))

        cfg_base = load_config().get("base_equity", -1)
        base_equity = svc.get_info("equity") if cfg_base == -1 else cfg_base
        log.info("Base equity: %.2f (source: %s)", base_equity,
                 "account_info" if cfg_base == -1 else "config.json")

        while True:
            try:
                df = fetch_df(symbol, timeframe)
                bar_time = df.index[-2]

                if not is_enabled():
                    log.info("Bot disabled (is_enable=false) — skipping.")
                    time.sleep(POLL_SECONDS)
                    continue

                if bar_time == last_bar_time:
                    time.sleep(POLL_SECONDS)
                    continue

                last_bar_time = bar_time
                log.info("New bar: %s  close=%.5f", bar_time, df["close"].iloc[-2])

                equity = svc.get_info("equity")
                sig    = latest_signal(df, **long_params, **short_params)
                pos    = get_position(symbol)

                log.info("Signal: %s  |  Position: %s  |  Equity: %.2f  |  Lot: %.2f",
                         sig, pos, equity, calc_volume(equity, base_equity))

                # ── ฝั่ง Long ──────────────────────────────────────────────
                if sig["long_entry"] and pos != "BUY":
                    if pos == "SELL":
                        close_position(symbol)
                    open_order(svc, symbol, "BUY", sl_pct, tp_pct, equity, base_equity)

                elif sig["long_exit"] and pos == "BUY":
                    close_position(symbol)

                # ── ฝั่ง Short ─────────────────────────────────────────────
                if sig["short_entry"] and pos != "SELL":
                    if pos == "BUY":
                        close_position(symbol)
                    open_order(svc, symbol, "SELL", sl_pct, tp_pct, equity, base_equity)

                elif sig["short_exit"] and pos == "SELL":
                    close_position(symbol)

            except KeyboardInterrupt:
                log.info("Stopped by user.")
                line_send(f"🛑 Bot stopped\n{symbol} {tf_str}")
                break
            except Exception as exc:
                log.error("Error: %s", exc, exc_info=True)
                line_send(f"⚠️ Bot error\n{symbol} {tf_str}\n{exc}")
                time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_live()
