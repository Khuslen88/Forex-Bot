# Adaptive Forex Trading Bot

A reinforcement learning-based forex trading bot that combines **technical analysis**, **fundamental analysis**, and **sentiment analysis** to trade 5 major currency pairs. Trained with DQN and PPO agents, with built-in risk management (stop loss / take profit) and a live paper trading system.

## Results

Best agent per pair on the test period (20% holdout, ~2 years):

| Pair | Best Agent | Return | Sharpe Ratio | Win Rate | Max Drawdown | Profit Factor |
|------|-----------|--------|-------------|----------|-------------|--------------|
| USD/JPY | DQN | +28.17% | 1.398 | 55.2% | -7.3% | 1.623 |
| USD/CAD | PPO | +16.59% | 1.574 | 68.3% | -5.1% | 1.829 |
| GBP/USD | DQN | +14.31% | 1.077 | 58.2% | -5.5% | 1.748 |
| AUD/USD | DQN | +8.53% | 0.523 | 54.8% | -9.7% | 1.160 |
| EUR/USD | DQN | +5.05% | 0.417 | 54.4% | -10.7% | 1.097 |

- 3/5 pairs hit all 4 project targets (Sharpe > 1.0, DD < 20%, Win Rate > 45%, Profit Factor > 1.2)
- All 5 pairs profitable using the best agent
- All pairs beat Buy-and-Hold, SMA Crossover, and Random Agent baselines

## Features

### Analysis Layers
- **Technical Analysis:** RSI, MACD, Bollinger Bands, ATR, ADX (trend strength), SMA crossover
- **Fundamental Analysis:** FRED economic data — Fed/ECB rate differential, US yield curve (10Y-2Y)
- **Sentiment Analysis:** RSS feed scraping from forex news sites, scored with TextBlob NLP

### RL Agents
- **DQN (Deep Q-Network):** Value-based RL, learns which action has the highest future reward
- **PPO (Proximal Policy Optimization):** Policy-based RL, learns a trading strategy directly
- Multi-seed training (best of 3) to reduce variance from random initialization

### Risk Management
- **Stop Loss:** 2% automatic position close to limit downside
- **Take Profit:** 3% automatic position close to lock in gains
- **Risk-adjusted reward function:** Sharpe-inspired with drawdown penalty

### Live Trading
- **Paper trading:** Simulated trading with real Yahoo Finance prices (no API key needed)
- **OANDA integration:** REST API wrapper for live demo account trading
- **Trade logging:** All decisions saved to CSV with timestamps
- **Streamlit dashboard:** Real-time signals, positions, P&L, and charts

## Project Structure

```
forex-bot/
├── config/
│   └── settings.py          # All hyperparameters, pairs, paths
├── data/
│   ├── raw/                  # 10yr daily OHLCV (13,115 candles)
│   ├── processed/            # Feature-engineered datasets
│   └── economic/             # FRED macro data
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py      # DQN training/inference
│   │   ├── ppo_agent.py      # PPO training/inference
│   │   └── baselines.py      # Buy-and-hold, SMA, Random
│   ├── environment/
│   │   ├── forex_env.py      # Base Gym environment (SL/TP, reward)
│   │   └── feature_env.py    # Extended env with indicators + sentiment
│   ├── features/
│   │   ├── indicators.py     # Technical indicators + FRED merge
│   │   └── sentiment.py      # News sentiment analysis
│   ├── backtesting/
│   │   └── evaluate.py       # Metrics, walk-forward validation
│   └── broker/
│       ├── paper_client.py   # Paper trading (Yahoo Finance)
│       └── oanda_client.py   # OANDA REST API wrapper
├── models/                   # Saved .zip model files
├── demo.py                   # Full training + evaluation pipeline
├── live_trader.py            # Live/paper trading script
├── dashboard.py              # Streamlit dashboard
└── requirements.txt
```

## Quick Start

### Setup
```bash
git clone <repo-url>
cd forex-bot
pip install -r requirements.txt
```

### Train a model
```bash
# DQN (default)
python demo.py --pair EURUSD --seeds 3

# PPO
python demo.py --pair EURUSD --agent ppo --seeds 3

# Quick test (20k steps)
python demo.py --pair EURUSD --quick
```

### Load and evaluate a pre-trained model
```bash
python demo.py --pair EURUSD --load models/dqn_feature_EURUSD.zip
```

### Paper trading
```bash
# See model signal without trading
python live_trader.py --pair all --dry-run

# Execute trades on paper account
python live_trader.py --pair all

# Check account status
python live_trader.py --status

# Run daily in a loop
python live_trader.py --pair all --loop
```

### Dashboard
```bash
streamlit run dashboard.py
```

## Technical Approach

### Observation Space (35 features)
| Category | Features | Count |
|----------|---------|-------|
| Price returns | Daily returns over 20-day window | 20 |
| Volatility | Annualized rolling volatility | 1 |
| Position | Current position (-1/0/1) | 1 |
| Technical | RSI, MACD, MACD signal, BB width, ATR, ADX, SMA ratio | 7 |
| Fundamental | Rate differential, yield curve, rate diff MA | 3 |
| Sentiment | Sentiment score, sentiment volatility, news volume | 3 |

### Action Space
- **0 = Flat** (no position)
- **1 = Long** (buy — profit when price rises)
- **2 = Short** (sell — profit when price falls)

### Reward Function
1. **Risk-adjusted PnL** — daily PnL divided by rolling volatility (Sharpe-like)
2. **Drawdown penalty** — penalizes being far below peak equity
3. **Trade completion bonus** — rewards profitable trade closures
4. **SL/TP shaping** — bonus for take-profit hits, mild penalty for stop-loss
5. **Flat penalty** — discourages prolonged inactivity

### Validation
- **Train/test split:** 80/20 (first 8 years train, last 2 years test)
- **Walk-forward validation:** 5-fold time-series cross-validation
- **Baselines compared:** Buy-and-Hold, SMA Crossover (50/200), Random Agent

## Limitations

- **Daily timeframe only** — trained on daily candles; doesn't capture intraday moves
- **Single pair at a time** — no portfolio-level correlation management
- **Historical sentiment proxy** — live sentiment uses RSS feeds, but training uses price-based proxy since historical headlines aren't available
- **Market regime sensitivity** — performs better in trending markets (high ADX); struggles in choppy/sideways conditions
- **No live money tested** — paper trading only; would need slippage modeling for real deployment

## Next Steps

- Intraday support (1H/4H candles) using MT5 historical data export
- Ensemble voting across multiple models per pair
- Portfolio-level risk management (pair correlation awareness)
- Real-time sentiment from financial news APIs
- Automated retraining pipeline (monthly refresh on new data)

## Tech Stack

- **RL Framework:** Stable-Baselines3 (PyTorch)
- **Environment:** Gymnasium (custom)
- **Data:** Yahoo Finance, FRED API
- **Indicators:** ta (Technical Analysis library)
- **Sentiment:** TextBlob NLP
- **Dashboard:** Streamlit + Plotly
- **Language:** Python 3.9

## Author

Data Science Capstone Project — Spring 2026
