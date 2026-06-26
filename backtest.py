"""
Backtest with vectorbt using the shared signal function.

Install: pip install vectorbt
"""

import MetaTrader5 as mt5
import pandas as pd
import vectorbt as vbt

from app import MT5Service
from app.signals import generate_signals

SYMBOL      = "EURUSDm"
TIMEFRAME   = mt5.TIMEFRAME_H4
NUM_BARS    = 5000
INIT_CASH   = 2_000.0
TRAIN_RATIO = 0.70      # 70% train / 30% out-of-sample test


def fetch_ohlcv(svc: MT5Service, symbol: str, timeframe, num_bars: int) -> pd.DataFrame:
    return svc.copy_rates(symbol, timeframe, num_bars=num_bars)


def split_data(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    split = int(len(df) * train_ratio)
    return df.iloc[:split], df.iloc[split:]


def run_backtest(df: pd.DataFrame, long_params: dict = None, short_params: dict = None):
    long_params  = long_params  or {}
    short_params = short_params or {}
    signals = generate_signals(
        df,
        **long_params,
        rsi_period_short=short_params.get("rsi_period"),
        ema_fast_short=short_params.get("ema_fast"),
    )

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=signals["entries"],
        exits=signals["exits"],
        short_entries=signals["short_entries"],
        short_exits=signals["short_exits"],
        init_cash=INIT_CASH,
        fees=0.00001,   # 0.1 pip/side = 0.2 pip spread round-trip (Standard)
        sl_stop=0.01,       # 1% stop loss
        tp_stop=0.02,       # 2% take profit
        freq="4h",
    )

    print("\n" + "=" * 45)
    print("           BACKTEST RESULTS              ")
    print("=" * 45)
    stats = pf.stats()
    print(stats)
    print(f"\nTotal Trades: {int(stats['Total Trades'])}")

    pf.plot().show()
    return pf


def optimize_short(df: pd.DataFrame):
    """Parameter sweep บน short side เท่านั้น."""
    import itertools

    combos = list(itertools.product(range(10, 25), range(5, 15)))

    se_cols, sx_cols, labels = [], [], []
    for rsi_p, ema_f in combos:
        sig = generate_signals(df, rsi_period_short=rsi_p, ema_fast_short=ema_f)
        if sig["short_entries"].sum() == 0:
            continue
        label = f"rsi{rsi_p}_emaf{ema_f}"
        se_cols.append(sig["short_entries"].rename(label))
        sx_cols.append(sig["short_exits"].rename(label))
        labels.append((rsi_p, ema_f))

    if not labels:
        print("No valid short combinations found.")
        return {}

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=pd.concat([s * False for s in se_cols], axis=1),
        exits=pd.concat([s * False for s in sx_cols], axis=1),
        short_entries=pd.concat(se_cols, axis=1),
        short_exits=pd.concat(sx_cols, axis=1),
        init_cash=INIT_CASH,
        fees=0.00001,
        sl_stop=0.01,
        tp_stop=0.02,
        freq="4h",
    )

    sharpe = pf.sharpe_ratio()
    rsi_p, ema_f = labels[sharpe.argmax()]
    best_params = {"rsi_period": rsi_p, "ema_fast": ema_f}

    print("\n[Short] Best params:", best_params)
    print(f"[Short] Best Sharpe: {sharpe.max():.4f}")
    return best_params


def optimize(df: pd.DataFrame):
    """Parameter sweep: ทดสอบหลาย RSI period พร้อมกันด้วย vectorbt."""
    import itertools

    combos = list(itertools.product(range(10, 25), range(5, 15)))

    entries_cols, exits_cols, se_cols, sx_cols, labels = [], [], [], [], []
    for rsi_p, ema_f in combos:
        sig = generate_signals(df, rsi_period=rsi_p, ema_fast=ema_f)
        if sig["entries"].sum() == 0 and sig["short_entries"].sum() == 0:
            continue
        label = f"rsi{rsi_p}_emaf{ema_f}"
        entries_cols.append(sig["entries"].rename(label))
        exits_cols.append(sig["exits"].rename(label))
        se_cols.append(sig["short_entries"].rename(label))
        sx_cols.append(sig["short_exits"].rename(label))
        labels.append((rsi_p, ema_f))

    if not labels:
        print("No valid combinations found (all produced 0 entries).")
        return {}

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=pd.concat(entries_cols, axis=1),
        exits=pd.concat(exits_cols, axis=1),
        short_entries=pd.concat(se_cols, axis=1),
        short_exits=pd.concat(sx_cols, axis=1),
        init_cash=INIT_CASH,
        fees=0.00001,   # 0.1 pip/side = 0.2 pip spread round-trip (Standard)
        sl_stop=0.01,
        tp_stop=0.02,
        freq="4h",
    )

    sharpe = pf.sharpe_ratio()
    rsi_p, ema_f = labels[sharpe.argmax()]
    best_params = {"rsi_period": rsi_p, "ema_fast": ema_f}

    print("\nBest params:", best_params)
    print(f"Best Sharpe: {sharpe.max():.4f}")
    return best_params


if __name__ == "__main__":
    with MT5Service() as svc:
        df = fetch_ohlcv(svc, SYMBOL, TIMEFRAME, NUM_BARS)

    train_df, test_df = split_data(df)
    print(f"Loaded {len(df)} bars for {SYMBOL}")
    print(f"Train : {train_df.index[0]} -> {train_df.index[-1]} ({len(train_df)} bars)")
    print(f"Test  : {test_df.index[0]} -> {test_df.index[-1]} ({len(test_df)} bars)")

    print("\n--- IN-SAMPLE OPTIMIZATION: LONG (train set) ---")
    best_long = optimize(train_df)

    print("\n--- IN-SAMPLE OPTIMIZATION: SHORT (train set) ---")
    best_short = optimize_short(train_df)

    print(f"\nLong  best params : {best_long}")
    print(f"Short best params : {best_short}")

    if best_long:
        print("\n--- IN-SAMPLE BACKTEST (train set) ---")
        run_backtest(train_df, long_params=best_long, short_params=best_short)

        print("\n--- OUT-OF-SAMPLE BACKTEST (test set) ---")
        run_backtest(test_df, long_params=best_long, short_params=best_short)
