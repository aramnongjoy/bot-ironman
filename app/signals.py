import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def generate_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_momentum: float = 40.0,
    ema_fast: int = 9,
    ema_slow: int = 21,
    ema_trend: int = 50,
    rsi_period_short: int = None,
    ema_fast_short: int = None,
) -> pd.DataFrame:
    """Return a DataFrame with boolean 'entries' and 'exits' columns.

    Long/short sides can use independent rsi_period and ema_fast.
    If rsi_period_short / ema_fast_short are None, they fall back to the long values.

    Parameters
    ----------
    df : must have a 'close' column (lowercase)
    """
    close = df["close"]

    _rsi_p_s = rsi_period_short if rsi_period_short is not None else rsi_period
    _ema_f_s = ema_fast_short   if ema_fast_short   is not None else ema_fast

    rsi_l   = compute_rsi(close, rsi_period)
    rsi_s   = compute_rsi(close, _rsi_p_s)
    ema_fl  = compute_ema(close, ema_fast)
    ema_fs  = compute_ema(close, _ema_f_s)
    ema_slow_line = compute_ema(close, ema_slow)
    ema_t   = compute_ema(close, ema_trend)

    trend_up   = close > ema_t
    trend_down = close < ema_t

    long_cross_up  = (ema_fl > ema_slow_line) & (ema_fl.shift(1) <= ema_slow_line.shift(1))
    long_cross_dn  = (ema_fl < ema_slow_line) & (ema_fl.shift(1) >= ema_slow_line.shift(1))
    short_cross_dn = (ema_fs < ema_slow_line) & (ema_fs.shift(1) >= ema_slow_line.shift(1))
    short_cross_up = (ema_fs > ema_slow_line) & (ema_fs.shift(1) <= ema_slow_line.shift(1))

    entries       = long_cross_up  & (rsi_l > rsi_momentum)         & trend_up
    exits         = long_cross_dn  | (rsi_l > rsi_overbought)
    short_entries = short_cross_dn & (rsi_s < (100 - rsi_momentum)) & trend_down
    short_exits   = short_cross_up | (rsi_s < (100 - rsi_overbought))

    return pd.DataFrame(
        {"entries": entries, "exits": exits,
         "short_entries": short_entries, "short_exits": short_exits},
        index=df.index,
    )


def latest_signal(df: pd.DataFrame, **kwargs) -> dict:
    """Return dict of the 4 boolean signals on the last CLOSED bar.

    Keys: long_entry, long_exit, short_entry, short_exit
    """
    signals = generate_signals(df, **kwargs)
    last = signals.iloc[-2]   # -2 = last closed bar (not the forming bar)
    return {
        "long_entry":  bool(last["entries"]),
        "long_exit":   bool(last["exits"]),
        "short_entry": bool(last["short_entries"]),
        "short_exit":  bool(last["short_exits"]),
    }
