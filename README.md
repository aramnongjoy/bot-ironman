# Bot Ironman

Automated trading bot for MetaTrader 5 using EMA Crossover + RSI momentum strategy with LINE Messenger notifications.

## Trading Setup

| Setting | Value |
|---|---|
| Symbol | EURUSDm (micro lot) |
| Timeframe | H1 |
| Stop Loss | 1% |
| Take Profit | 2% |
| Risk/Reward | 1 : 2 |
| Direction | Long & Short |

## Strategy

**Entry conditions (Long)**
- EMA fast crosses above EMA slow
- RSI > 40 (momentum filter)
- Price > EMA trend (trend filter)

**Exit conditions (Long)**
- EMA fast crosses below EMA slow, or
- RSI > 70 (overbought)

**Short side** uses mirror-image conditions.

Default indicators: EMA 10/21/50, RSI 14 — optimized via walk-forward cross-validation.

## Features

- **Live trading** — connects to MT5, polls every 5 minutes, acts only on closed bars (no repainting)
- **Long & Short** — trades both directions simultaneously
- **Config-driven** — symbol, timeframe, indicator params, SL/TP, and lot size all read from `config.json`; no code changes needed to switch markets
- **Dynamic lot sizing** — scales position size proportionally to current equity
- **Magic number isolation** — only manages positions opened by this bot; never touches manual trades
- **LINE notifications** — sends messages on bot start, order open/close, error, and stop
- **Walk-forward CV optimization** — selects best RSI/EMA parameters using 4-fold cross-validation to reduce overfitting
- **Backtest** — powered by vectorbt; splits data 70/30 train/test and outputs HTML report with equity curves

## Backtest Results (EURUSDm H1)

| | In-Sample (70%) | Out-of-Sample (30%) |
|---|---|---|
| Sharpe Ratio | 0.86 | **1.75** |
| Total Return | 3.82% | 2.95% |
| Max Drawdown | 4.11% | 1.90% |
| Win Rate | 38.8% | 41.9% |
| Profit Factor | 1.13 | 1.29 |
| Trades | 245 | 105 |

Data: 2024-11-21 → 2026-07-03 (10,000 bars)

## Project Structure

```
bot-ironman/
├── live.py              # Live trading loop
├── backtest.py          # Backtest & parameter optimization
├── backtest.html        # Latest backtest report
├── config.json          # Runtime configuration
├── .env                 # MT5 credentials & LINE API keys
└── app/
    ├── signals.py       # Signal generation (EMA crossover + RSI)
    ├── mt5_service.py   # MT5 connection & order management
    └── line_notify.py   # LINE Messenger notifications
```

## Configuration (`config.json`)

```json
{
  "is_enable": true,
  "symbol": "EURUSDm",
  "timeframe": "H1",
  "lot_size_type": "dynamic",
  "lot_size": 0.01,
  "sl_pct": 0.01,
  "tp_pct": 0.02,
  "long_params":  { "rsi_period": 14, "ema_fast": 10, "ema_slow": 21, "ema_trend": 50 },
  "short_params": { "rsi_period_short": 14, "ema_fast_short": 11 }
}
```

Set `"is_enable": false` to pause the bot without stopping the process.

## Environment Variables (`.env`)

```
MT5_LOGIN=...
MT5_PASSWORD=...
MT5_SERVER=...
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

LINE_CHANNEL_ACCESS_TOKEN=...
LINE_USER_ID=...
```

## Usage

```bash
# Run live bot
python live.py

# Run backtest & generate backtest.html
python backtest.py
```

## Requirements

```bash
pip install MetaTrader5 vectorbt pandas numpy python-dotenv requests
```
