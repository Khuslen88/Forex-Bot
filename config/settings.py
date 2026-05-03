"""
Forex Bot Configuration Settings
"""

# =============================================================================
# TRADING PAIRS
# =============================================================================
FOREX_PAIRS = [
    'EURUSD',
    'USDJPY',
    'GBPUSD',
    'AUDUSD',
    'USDCAD',
]

YAHOO_SYMBOLS = {
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'USDJPY=X',
    'GBPUSD': 'GBPUSD=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'USDCAD=X',
}

# =============================================================================
# DATA SETTINGS
# =============================================================================
DATA_START_DATE = '2016-01-01'
DATA_END_DATE = '2026-01-31'
TIMEFRAMES = ['1h', '4h', '1d']

# =============================================================================
# TRADING SETTINGS
# =============================================================================
INITIAL_BALANCE = 10000
RISK_PER_TRADE = 0.02  # 2% risk per trade
STOP_LOSS_PCT = 0.02   # 2% stop loss from entry price
TAKE_PROFIT_PCT = 0.03 # 3% take profit from entry price
SPREAD_PIPS = {
    'EURUSD': 1.0,
    'USDJPY': 1.2,
    'GBPUSD': 1.5,
    'AUDUSD': 1.3,
    'USDCAD': 1.8,
}

# =============================================================================
# RL SETTINGS
# =============================================================================
# DQN Hyperparameters
DQN_CONFIG = {
    'learning_rate': 1e-4,              # lower LR for stable convergence
    'buffer_size': 200_000,             # larger replay buffer
    'learning_starts': 2000,            # more random exploration before learning
    'batch_size': 128,                  # larger batches reduce gradient noise
    'gamma': 0.99,                      # longer horizon — forex trends persist
    'exploration_fraction': 0.4,        # explore for 40% of training
    'exploration_final_eps': 0.03,      # less random noise at convergence
    'target_update_interval': 1000,     # slower target net updates for stability
    'max_grad_norm': 5.0,              # gradient clipping
}

# Network architecture — bigger than default 64x64
DQN_NET_ARCH = [256, 128]

# PPO Hyperparameters
PPO_CONFIG = {
    'learning_rate': 3e-4,              # standard PPO LR (with linear schedule)
    'n_steps': 2048,                    # rollout length per update
    'batch_size': 64,                   # minibatch size for SGD
    'n_epochs': 10,                     # optimization epochs per rollout
    'gamma': 0.99,                      # discount factor
    'clip_range': 0.2,                  # PPO clipping parameter
    'ent_coef': 0.01,                   # entropy bonus — encourage exploration
    'max_grad_norm': 0.5,               # gradient clipping
}

PPO_NET_ARCH = [256, 128]

# Training
TOTAL_TIMESTEPS = 500_000
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# =============================================================================
# 1H TIMEFRAME OVERRIDES
# =============================================================================
# 1H candles move ~5x less than daily candles, so SL/TP must be tighter and
# minimum hold duration prevents the agent from churning the position every bar.
# More candles also means more training timesteps to converge.
TIMEFRAME_CONFIGS = {
    '1d': {
        'stop_loss_pct':   0.02,
        'take_profit_pct': 0.03,
        'min_hold_steps':  1,
        'total_timesteps': 500_000,
        'data_file':       'forex_dataset_daily.csv',
    },
    '1h': {
        'stop_loss_pct':   0.005,   # 0.5% — typical 1H ATR is ~0.1-0.3%
        'take_profit_pct': 0.008,   # 0.8% — keep ~1.6 reward:risk
        'min_hold_steps':  4,       # hold at least 4 hours before exiting
        'total_timesteps': 1_500_000,
        'data_file':       'forex_dataset_1h.csv',
    },
}

# =============================================================================
# FEATURE SETTINGS
# =============================================================================
TECHNICAL_INDICATORS = [
    'RSI',
    'MACD',
    'MACD_signal',
    'MACD_hist',
    'BB_upper',
    'BB_middle',
    'BB_lower',
    'ATR',
    'SMA_50',
    'SMA_200',
]

LOOKBACK_PERIOD = 30  # Days of history for state

# =============================================================================
# EVALUATION METRICS
# =============================================================================
TARGETS = {
    'sharpe_ratio': 1.0,
    'max_drawdown': 0.20,  # 20%
    'win_rate': 0.45,      # 45%
    'profit_factor': 1.2,
}

# =============================================================================
# PATHS
# =============================================================================
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_ECONOMIC_PATH = os.path.join(PROJECT_ROOT, 'data', 'economic')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')

LIVE_TRADE_UNITS = 1000  # micro lot (safe for demo account)
