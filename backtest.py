"""
Backtest with vectorbt using the shared signal function.

Install: pip install vectorbt
"""

import MetaTrader5 as mt5
import pandas as pd
import vectorbt as vbt

from app import MT5Service
from app.signals import generate_signals

SYMBOL      = "BTCUSDc"
TIMEFRAME   = mt5.TIMEFRAME_H4
NUM_BARS    = 5000
INIT_CASH   = 2_154.21
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

    combos = list(itertools.product(range(10, 25), range(3, 15)))

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

    combos = list(itertools.product(range(10, 25), range(3, 15)))

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


def _run_sweep(df: pd.DataFrame, combos, signal_kwargs_fn):
    """Helper: build signal columns and run one portfolio sweep. Returns (pf, valid_combos)."""
    entries_cols, exits_cols, se_cols, sx_cols, valid = [], [], [], [], []
    for combo in combos:
        sig = generate_signals(df, **signal_kwargs_fn(combo))
        if sig["entries"].sum() == 0 and sig["short_entries"].sum() == 0:
            continue
        label = "_".join(str(v) for v in combo)
        entries_cols.append(sig["entries"].rename(label))
        exits_cols.append(sig["exits"].rename(label))
        se_cols.append(sig["short_entries"].rename(label))
        sx_cols.append(sig["short_exits"].rename(label))
        valid.append(combo)
    if not valid:
        return None, []
    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=pd.concat(entries_cols, axis=1),
        exits=pd.concat(exits_cols, axis=1),
        short_entries=pd.concat(se_cols, axis=1),
        short_exits=pd.concat(sx_cols, axis=1),
        init_cash=INIT_CASH,
        fees=0.00001,
        sl_stop=0.01,
        tp_stop=0.02,
        freq="4h",
    )
    return pf, valid


def optimize_cv(df: pd.DataFrame, n_folds: int = 4, top_n: int = 10):
    """Walk-forward cross-validation: แบ่ง df เป็น n_folds validation windows
    แล้วเลือก params ที่ได้ avg Sharpe สูงสุดข้าม folds ทั้งหมด."""
    import itertools
    import numpy as np

    combos = list(itertools.product(range(10, 25), range(3, 15)))

    def kwargs_fn(c):
        rsi_p, ema_f = c
        return {"rsi_period": rsi_p, "ema_fast": ema_f}

    n_total = len(df)
    val_size = n_total // (n_folds + 1)
    cv_scores = {c: [] for c in combos}

    print(f"  Running {n_folds}-fold walk-forward CV ({val_size} bars/fold)...")
    for fold_i in range(n_folds):
        val_start = val_size * (fold_i + 1)
        val_end = min(val_start + val_size, n_total)
        val_df = df.iloc[val_start:val_end]

        pf, valid = _run_sweep(val_df, combos, kwargs_fn)
        fold_sharpes = {}
        if pf is not None:
            sharpes = pf.sharpe_ratio()
            for i, combo in enumerate(valid):
                fold_sharpes[combo] = float(sharpes.iloc[i])

        for combo in combos:
            cv_scores[combo].append(fold_sharpes.get(combo, np.nan))

        print(f"    Fold {fold_i+1}: {val_df.index[0].date()} -> {val_df.index[-1].date()} "
              f"| {len(valid)}/{len(combos)} combos had trades")

    avg_cv = {c: np.nanmean(v) for c, v in cv_scores.items()}
    avg_cv = {c: v for c, v in avg_cv.items() if not np.isnan(v)}
    if not avg_cv:
        print("No valid combinations found.")
        return {}

    sorted_combos = sorted(avg_cv.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*55}")
    print(f"  TOP {top_n} by avg CV Sharpe ({n_folds} validation windows)")
    print(f"{'='*55}")
    print(f"{'#':>2}  {'RSI':>4}  {'EMAf':>5}  {'CV Sharpe':>10}")
    print("-" * 30)
    for rank, ((rsi_p, ema_f), cv_sharpe) in enumerate(sorted_combos[:top_n], 1):
        print(f"{rank:>2}  {rsi_p:>4}  {ema_f:>5}  {cv_sharpe:>10.4f}")
    print("=" * 55)

    (rsi_p, ema_f), best_cv = sorted_combos[0]
    best_params = {"rsi_period": rsi_p, "ema_fast": ema_f}
    print(f"\n[CV] Best params: {best_params}  (avg CV Sharpe: {best_cv:.4f})")
    return best_params


def optimize_short_cv(df: pd.DataFrame, n_folds: int = 4):
    """Walk-forward CV สำหรับ short side."""
    import itertools
    import numpy as np

    combos = list(itertools.product(range(10, 25), range(3, 15)))

    n_total = len(df)
    val_size = n_total // (n_folds + 1)
    cv_scores = {c: [] for c in combos}

    print(f"  Running {n_folds}-fold walk-forward CV for short side...")
    for fold_i in range(n_folds):
        val_start = val_size * (fold_i + 1)
        val_end = min(val_start + val_size, n_total)
        val_df = df.iloc[val_start:val_end]

        entries_cols, exits_cols, se_cols, sx_cols, valid = [], [], [], [], []
        for rsi_p, ema_f in combos:
            sig = generate_signals(val_df, rsi_period_short=rsi_p, ema_fast_short=ema_f)
            if sig["short_entries"].sum() == 0:
                continue
            label = f"{rsi_p}_{ema_f}"
            entries_cols.append((sig["entries"] * False).rename(label))
            exits_cols.append((sig["exits"] * False).rename(label))
            se_cols.append(sig["short_entries"].rename(label))
            sx_cols.append(sig["short_exits"].rename(label))
            valid.append((rsi_p, ema_f))

        fold_sharpes = {}
        if valid:
            pf = vbt.Portfolio.from_signals(
                close=val_df["close"],
                entries=pd.concat(entries_cols, axis=1),
                exits=pd.concat(exits_cols, axis=1),
                short_entries=pd.concat(se_cols, axis=1),
                short_exits=pd.concat(sx_cols, axis=1),
                init_cash=INIT_CASH,
                fees=0.00001,
                sl_stop=0.01,
                tp_stop=0.02,
                freq="4h",
            )
            sharpes = pf.sharpe_ratio()
            for i, combo in enumerate(valid):
                fold_sharpes[combo] = float(sharpes.iloc[i])

        for combo in combos:
            cv_scores[combo].append(fold_sharpes.get(combo, np.nan))

        print(f"    Fold {fold_i+1}: {val_df.index[0].date()} -> {val_df.index[-1].date()} "
              f"| {len(valid)}/{len(combos)} combos had trades")

    avg_cv = {c: np.nanmean(v) for c, v in cv_scores.items()}
    avg_cv = {c: v for c, v in avg_cv.items() if not np.isnan(v)}
    if not avg_cv:
        print("No valid short combinations found.")
        return {}

    best_combo = max(avg_cv, key=avg_cv.get)
    rsi_p, ema_f = best_combo
    best_params = {"rsi_period": rsi_p, "ema_fast": ema_f}
    print(f"\n[Short CV] Best params: {best_params}  (avg CV Sharpe: {avg_cv[best_combo]:.4f})")
    return best_params


def compare_params(df: pd.DataFrame, top_n: int = 10):
    """แสดงตารางเปรียบเทียบทุก combination เรียงตาม Sharpe Ratio."""
    import itertools

    combos = list(itertools.product(range(10, 25), range(3, 15)))

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
        print("No valid combinations found.")
        return

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=pd.concat(entries_cols, axis=1),
        exits=pd.concat(exits_cols, axis=1),
        short_entries=pd.concat(se_cols, axis=1),
        short_exits=pd.concat(sx_cols, axis=1),
        init_cash=INIT_CASH,
        fees=0.00001,
        sl_stop=0.01,
        tp_stop=0.02,
        freq="4h",
    )

    all_stats = pf.stats(agg_func=None)

    results = pd.DataFrame({
        "rsi_period":    [r for r, _ in labels],
        "ema_fast":      [f for _, f in labels],
        "sharpe":        pf.sharpe_ratio().values,
        "return_pct":    pf.total_return().values * 100,
        "max_dd_pct":    pf.max_drawdown().values * 100,
        "win_rate":      all_stats["Win Rate [%]"].values,
        "total_trades":  all_stats["Total Trades"].values,
        "profit_factor": all_stats["Profit Factor"].values,
    }).sort_values("sharpe", ascending=False).reset_index(drop=True)

    print(f"\n{'='*75}")
    print(f"  TOP {top_n} PARAMETER COMBINATIONS (sorted by Sharpe)")
    print(f"{'='*75}")
    print(f"{'#':>2}  {'RSI':>4}  {'EMA':>4}  {'Sharpe':>7}  {'Return%':>8}  "
          f"{'MaxDD%':>7}  {'WinRate%':>9}  {'Trades':>7}  {'PF':>6}")
    print("-" * 75)
    for i, row in results.head(top_n).iterrows():
        print(f"{i+1:>2}  {int(row.rsi_period):>4}  {int(row.ema_fast):>4}  "
              f"{row.sharpe:>7.4f}  {row.return_pct:>8.2f}  "
              f"{row.max_dd_pct:>7.2f}  {row.win_rate:>9.1f}  "
              f"{int(row.total_trades):>7}  {row.profit_factor:>6.3f}")
    print("=" * 75)
    return results


if __name__ == "__main__":
    with MT5Service() as svc:
        df = fetch_ohlcv(svc, SYMBOL, TIMEFRAME, NUM_BARS)

    train_df, test_df = split_data(df)
    print(f"Loaded {len(df)} bars for {SYMBOL}")
    print(f"Train : {train_df.index[0]} -> {train_df.index[-1]} ({len(train_df)} bars)")
    print(f"Test  : {test_df.index[0]} -> {test_df.index[-1]} ({len(test_df)} bars)")

    print("\n--- PARAMETER COMPARISON (in-sample, train set) ---")
    compare_params(train_df)

    print("\n--- WALK-FORWARD CV: LONG (train set, 4 folds) ---")
    best_long = optimize_cv(train_df, n_folds=4)

    print("\n--- WALK-FORWARD CV: SHORT (train set, 4 folds) ---")
    best_short = optimize_short_cv(train_df, n_folds=4)

    print(f"\nLong  best params (CV) : {best_long}")
    print(f"Short best params (CV) : {best_short}")

    if best_long:
        print("\n--- IN-SAMPLE BACKTEST (train set) ---")
        run_backtest(train_df, long_params=best_long, short_params=best_short)

        print("\n--- OUT-OF-SAMPLE BACKTEST (test set) ---")
        run_backtest(test_df, long_params=best_long, short_params=best_short)
