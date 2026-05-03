#!/bin/bash
# Train all 5 pairs x 2 agents on 1H data with 3 seeds each.
# Best model per (pair, agent) saved to models/{dqn,ppo}_feature_{PAIR}_1h.zip
# Estimated total time: ~12-16 hours on CPU.

set -e
cd "$(dirname "$0")"
source venv/bin/activate

PAIRS="EURUSD USDJPY GBPUSD AUDUSD USDCAD"
AGENTS="dqn ppo"
SEEDS=3

START=$(date +%s)
LOG_DIR="logs/1h_training"
mkdir -p "$LOG_DIR"

echo "==========================================="
echo "1H FULL TRAINING — started $(date)"
echo "Pairs:  $PAIRS"
echo "Agents: $AGENTS"
echo "Seeds:  $SEEDS"
echo "==========================================="

for PAIR in $PAIRS; do
  for AGENT in $AGENTS; do
    RUN_LOG="$LOG_DIR/${AGENT}_${PAIR}_1h.log"
    echo ""
    echo ">>> START $AGENT $PAIR 1h  ($(date +%H:%M:%S))"
    if python demo.py --pair "$PAIR" --agent "$AGENT" --timeframe 1h --seeds "$SEEDS" \
         > "$RUN_LOG" 2>&1; then
      # Extract best-seed sharpe from the log
      BEST=$(grep -E "Best seed:" "$RUN_LOG" | tail -1 || echo "n/a")
      echo "<<< DONE  $AGENT $PAIR 1h  ($(date +%H:%M:%S))  $BEST"
    else
      echo "<<< FAIL  $AGENT $PAIR 1h  ($(date +%H:%M:%S))  see $RUN_LOG"
    fi
  done
done

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))
echo ""
echo "==========================================="
echo "ALL TRAINING COMPLETE — total ${ELAPSED} min"
echo "Models saved to: models/{dqn,ppo}_feature_*_1h.zip"
echo "Per-run logs:    $LOG_DIR/"
echo "==========================================="
