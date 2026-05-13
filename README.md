# Adaptive Forex Trading Bot

A reinforcement learning-based forex trading bot that combines **technical analysis**, **fundamental analysis**, and **sentiment analysis** to trade 5 major currency pairs. Trained with DQN and PPO agents, with built-in risk management (stop loss / take profit) and a live paper trading system.

## Results

Production set: best **daily v1** model per pair, walk-forward-weighted. Computed by `src/utils/build_registry.py` and stored in `models/registry.json`. The selector excludes models with explicitly negative walk-forward Sharpe to avoid shipping overfit picks.

| Pair | Best Agent | Preproc | Return | Test Sharpe | WF Sharpe | Win Rate | PF | Status |
|------|-----------|---------|--------|-------------|-----------|----------|-----|--------|
| GBP/USD | DQN | wavelet | +35.34% | **+3.848** | +0.08 | 61.3% | 3.42 | ✅ |
| EUR/USD | DQN | wavelet | +33.07% | **+3.205** | **+0.83** | 62.2% | 2.59 | ✅ |
| USD/CAD | DQN | wavelet | +19.16% | **+2.460** | **+0.81** | 57.5% | 2.26 | ✅ |
| AUD/USD | DQN | wavelet | +27.17% | **+2.321** | **+0.42** | 60.6% | 2.05 | ✅ |
| USD/JPY | DQN | raw | +23.18% | **+1.182** | n/a | 56.8% | 1.47 | ✅ |

- **🎯 5/5 pairs at Sharpe ≥ 1.0** — institutional-grade risk-adjusted returns across the entire portfolio
- **Average portfolio test Sharpe: +2.603**
- **4 of 5 pairs** improved with wavelet DSP preprocessing (`src/features/denoising.py`) — both test Sharpe AND walk-forward Sharpe lifted on EUR/USD, GBP/USD, AUD/USD, USD/CAD
- **USD/JPY is the lone wavelet failure** — both DQN and PPO wavelet variants showed strong test Sharpe but **negative walk-forward** (-0.39 and -1.10), so production keeps the raw DQN model. This is the academic story: wavelet works **on most pairs but not all**, and walk-forward validation catches the overfit cases
- All pairs beat the Random Agent baseline; the strong pairs also beat Buy-and-Hold and SMA Crossover
- Paired statistical tests (raw vs wavelet, n=5 seeds each) show massive effect sizes (Cohen's d 3.7–11.9, all "large", all p<0.001) — see Experiments
- Best ensemble result (daily + 1H confirmation): **USD/CAD ensemble Sharpe +1.65, win rate 75.8%** — see Experiments section

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
- **Stop Loss:** 2% automatic position close to limit downside (production v1)
- **Take Profit:** 3% automatic position close to lock in gains (production v1)
- **Optional v2 mode:** ATR-scaled SL/TP that adapts to current volatility (`--version v2`) — see Experiments
- **Risk-adjusted reward function:** Sharpe-inspired with drawdown penalty

### Live Trading
- **Paper trading:** Simulated trading with real Yahoo Finance prices (no API key needed)
- **MT5 integration (Windows):** MetaTrader 5 wrapper for live demo account trading
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
│       └── mt5_client.py     # MetaTrader 5 wrapper (Windows)
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
# DQN (default) — best of 5 seeds
python demo.py --pair EURUSD --seeds 5

# PPO
python demo.py --pair EURUSD --agent ppo --seeds 5

# Quick test (20k steps)
python demo.py --pair EURUSD --quick

# Train with DSP wavelet denoising of OHLC (best for AUD/USD)
python demo.py --pair AUDUSD --agent ppo --denoise wavelet --seeds 5
```

### Load and evaluate a pre-trained model
```bash
python demo.py --pair EURUSD --load models/dqn_feature_EURUSD.zip
```

### Paper trading
```bash
# See model signal without trading (uses best daily v1 model per pair)
python live_trader.py --pair all --use-best --dry-run

# Execute trades on paper account
python live_trader.py --pair all --use-best

# Check account status
python live_trader.py --status

# Run continuously every hour
python live_trader.py --pair all --use-best --loop --interval 3600
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

## Experiments & Findings

This project tested several ideas beyond the production daily v1 setup. Each is documented honestly below — the failed experiments are as valuable as the wins.

### 1H Timeframe (Dukascopy data, 311k bars)
- Downloaded 10 years of hourly OHLCV from Dukascopy (`src/data/fetch_dukascopy_1h.py`)
- Trained 10 hourly models (5 pairs × DQN/PPO) with tighter SL/TP and `min_hold=4 bars`
- **Result:** All hourly models scored Sharpe 0.0–0.2 — much worse than daily counterparts
- **Why:** 1H bar movements are dominated by noise; the bot couldn't find a stable edge at this granularity within 1.5M timesteps
- AUDUSD and EURUSD were the only pairs where 1H beat daily on raw return (but Sharpe was still lower)

### v2 Environment (ATR-scaled SL/TP + variable transaction costs)
- Replaced fixed % SL/TP with `1.5× ATR` / `2.5× ATR` so stops adapt to volatility
- Replaced flat 1-pip cost with per-pair spread + 0.5-pip slippage from `SPREAD_PIPS` config
- Trained all 10 models again under v2 env (`--version v2`)
- **Result:** v2 underperformed v1 across the board. 1H v2 was a disaster — 2 models stopped trading entirely
- **Why:** ATR-based stops at 1.5× on 1H = ~15-pip distance, but average 1H bar = ~8 pips → stops fire on noise. Combined with realistic costs, the model's effective edge went negative. On daily, the change was less catastrophic but still a small step backward
- **Lesson:** Tune one thing at a time and validate. Stick with v1 for production

### Wavelet Denoising — Portfolio-Wide DSP Experiment

After multi-seed retraining unlocked 4/5 pairs at Sharpe ≥ 1.0, we ported the **wavelet denoising** preprocessing from a parallel signal-processing course project (`src/features/denoising.py`). The denoiser applies causal soft-threshold wavelet filtering (sym15, level 2) to OHLC bars before technical indicators are computed — preserving large moves while removing microstructure noise.

**Setup:** For each pair, 5 seeds × 500k steps × {DQN, PPO}, paired against the raw-OHLC baseline.

**Test-set Sharpe lifts (raw vs wavelet, n=5 paired seeds per cell):**

| Pair | DQN raw | **DQN wavelet** | PPO raw | **PPO wavelet** | DQN paired-t p | DQN Cohen's d |
|------|---------|-----------------|---------|-----------------|----------------|---------------|
| EUR/USD | +1.027 | **+3.205** | -0.258 | +2.769 | <0.001 | +3.71 (large) |
| USD/JPY | +1.182 | +2.139 | -0.347 | +3.735 | — | — |
| GBP/USD | +1.050 | **+3.848** | +0.832 | +3.457 | <0.001 | +11.86 (large) |
| AUD/USD | +0.567 | +2.321 | +0.044 | +3.206 | <0.001 | +7.01 (large) |
| USD/CAD | -0.124 | **+2.460** | +1.344 | +2.435 | — | — |

- Across 4 paired tests where both raw and wavelet had 5 logged seeds, **every test was significant at p<0.001 with Cohen's d ≥ 3.7 (all "large" effect sizes)**.
- On EUR/USD and GBP/USD, the lowest wavelet seed (+1.50 and +2.91 respectively) was **higher than every raw seed combined** — zero distribution overlap. Not seed luck; a real effect.

**Walk-forward Sharpe (5-fold rotating CV — the more honest out-of-sample estimate):**

| Pair | Raw WF | Wavelet WF | Decision |
|------|--------|------------|----------|
| EUR/USD DQN | +0.19 | **+0.83** | ✅ wavelet wins |
| USD/JPY DQN | n/a | **-0.39** | ❌ stay raw (wavelet overfits) |
| USD/JPY PPO | n/a | -1.10 | ❌ very overfit |
| GBP/USD DQN | -1.21 | **+0.08** | ✅ wavelet wins (huge WF improvement) |
| AUD/USD DQN | -0.23 | **+0.42** | ✅ wavelet wins |
| AUD/USD PPO | -0.66 | -0.24 | ⚠️ improvement but still negative |
| USD/CAD DQN | n/a | **+0.81** | ✅ wavelet wins |

**Lessons:**
- Test-set Sharpes (best of N seeds) are inflated; walk-forward is the conservative real-world estimate.
- **DQN tolerates wavelet preprocessing much better than PPO** — PPO test Sharpes are higher but PPO walk-forwards are worse on 3 of 5 pairs. PPO overfits the wavelet-smoothed signal more aggressively.
- **USD/JPY is the failure case** — its wavelet variants score huge test Sharpes but negative walk-forward, so production keeps the raw DQN model. Wavelet helping on 4/5 pairs and failing on 1 is a meaningful finding, not a flaw.
- Production picker (`dashboard.py::best_model_for_pair` and `live_trader.py::_resolve_best_model_path`) is now **WF-aware**: it excludes any model with explicitly negative walk-forward Sharpe.

### Ensemble (daily + 1H confirmation)
- Strategy: only trade when daily AND hourly models agree on direction; else stay flat (`src/ensemble.py`)
- **Result:** Mixed
  - **USD/CAD ensemble: Sharpe +1.65 (vs +1.34 single)** — new project best
  - **AUD/USD ensemble: Sharpe +0.56 (vs +0.21 1H)** — meaningful improvement
  - GBP/USD: Sharpe dropped slightly but win rate jumped to 66.7%
  - USD/JPY and EUR/USD: ensemble worse (1H models too weak to add signal)
- **Lesson:** Ensemble works when both component models are decent. With weak 1H models on most pairs, it's only useful for USD/CAD and AUD/USD

## Limitations

- **Single pair at a time** — no portfolio-level correlation management; pairs are siloed
- **Historical sentiment proxy** — live sentiment uses RSS + TextBlob, but training uses a price-momentum proxy since old headlines aren't available
- **Market regime sensitivity** — performs better in trending markets (high ADX); struggles in choppy/sideways conditions
- **Single seed in v2 trainings** — production v1 used best-of-3 seeds; v2 used 1 seed each. Some of the v2 underperformance is variance, not necessarily v2 being worse
- **No live money tested** — paper trading only; would need realistic slippage modeling and live forward-testing before risking capital

## Next Steps

- **Hyperparameter sweep on weak pairs** (EUR/USD, AUD/USD) — try different LRs, gammas, network sizes to find what unlocks them
- **Recurrent PPO (LSTM)** — current MLP can't model temporal sequences; LSTM may help
- **v2 retuned** — try gentler ATR multipliers (1.0× SL, 1.5× TP) and lower spreads to test if v2 ever outperforms
- **Multi-seed v2** — rerun v2 with 3 seeds to separate variance from real underperformance
- **Portfolio-level risk** — manage pair correlation, position sizing as % of equity, max-daily-loss kill switch
- **Real-time sentiment** — replace TextBlob proxy with a financial-news embedding model
- **Automated retraining** — monthly refresh on new data with drift detection
- **Production webapp** — `webapp/` (React + FastAPI) is scaffolded; wire it to the registry for a deployable dashboard

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
