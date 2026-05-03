#!/bin/bash
# Train all 5 pairs x 2 agents on BOTH timeframes with v2 env.
# Two waves to fit 8 cores / 8 GB RAM:
#   Wave 1 (daily, 500k steps): 10 in parallel — ~30-45 min
#   Wave 2 (1H, 1.5M steps):    10 in parallel — ~60-90 min
# Total: ~2 hours wallclock.

cd "$(dirname "$0")"
source venv/bin/activate

LOG_DIR="logs/v2_training"
mkdir -p "$LOG_DIR"

# 1 thread per process so we get true core-level parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

START=$(date +%s)
echo "==========================================="
echo "v2 PARALLEL TRAINING — started $(date)"
echo "Wave 1 (daily) → Wave 2 (1H), 10 parallel each"
echo "==========================================="

run_wave() {
  local TF=$1
  local STEPS_NOTE=$2
  local -a PIDS

  echo ""
  echo "--- Wave: $TF ($STEPS_NOTE) ---"
  for PAIR in EURUSD USDJPY GBPUSD AUDUSD USDCAD; do
    for AGENT in dqn ppo; do
      LOG="$LOG_DIR/${AGENT}_${PAIR}_${TF}_v2.log"
      echo ">>> LAUNCH $AGENT $PAIR $TF v2  -> $LOG"
      python demo.py --pair "$PAIR" --agent "$AGENT" --timeframe "$TF" \
        --version v2 --seeds 1 > "$LOG" 2>&1 &
      PIDS+=($!)
    done
  done

  echo "Wave $TF: ${#PIDS[@]} jobs launched. Waiting ..."
  for PID in "${PIDS[@]}"; do
    wait "$PID"
    BEST=$(grep -E "Best seed|Sharpe Ratio" "$LOG_DIR"/*.log 2>/dev/null | tail -1)
    echo "<<< PID $PID exit ($(date +%H:%M:%S))"
  done
}

run_wave "1d" "500k steps"
echo ""
echo "=== Wave 1 (daily) complete at $(date +%H:%M:%S). Starting Wave 2 ==="
run_wave "1h" "1.5M steps"

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))
echo ""
echo "==========================================="
echo "ALL v2 TRAINING DONE — total ${ELAPSED} min"
echo "==========================================="
echo "Now run: python src/utils/build_registry.py"
echo "Then:    python src/ensemble.py --version v2"
