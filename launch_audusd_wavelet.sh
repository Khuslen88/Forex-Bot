#!/bin/bash
# AUD/USD wavelet retrain — same budget as Round 1 (5 seeds × 500k steps each)
# 2 parallel jobs: DQN + PPO. Estimated wallclock: ~2-3 hours.
#
# Models save as:
#   models/dqn_feature_AUDUSD_wavelet.zip
#   models/ppo_feature_AUDUSD_wavelet.zip
#
# Goal: push AUD/USD Sharpe over 1.0 → all 5 capstone pairs at institutional grade.

cd "$(dirname "$0")"
LOG_DIR="logs/wavelet_training"
mkdir -p "$LOG_DIR"

VENV_PY="$(pwd)/venv/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Launching AUD/USD wavelet trainings (DQN + PPO, detached) ..."
for AGENT in dqn ppo; do
  LOG="$LOG_DIR/${AGENT}_AUDUSD_wavelet.log"
  ( nohup "$VENV_PY" demo.py \
      --pair AUDUSD --agent "$AGENT" --timeframe 1d \
      --denoise wavelet --seeds 5 \
      > "$LOG" 2>&1 < /dev/null & )
  echo "  launched: $AGENT AUDUSD wavelet  -> $LOG"
done

echo ""
echo "Both launched and detached. Track:"
echo "  ls -la models/*_AUDUSD_wavelet.zip"
echo "  tail -f $LOG_DIR/dqn_AUDUSD_wavelet.log"
