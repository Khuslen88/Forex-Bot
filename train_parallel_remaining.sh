#!/bin/bash
# Launch the 6 remaining 1H trainings in parallel.
# (PPO USDJPY is assumed to still be running from the prior sequential script.)
# Total wall time: ~60-90 min (vs ~7h sequential).

cd "$(dirname "$0")"
source venv/bin/activate

LOG_DIR="logs/1h_training"
mkdir -p "$LOG_DIR"

START=$(date +%s)
echo "==========================================="
echo "PARALLEL 1H TRAINING — started $(date)"
echo "Launching 6 runs simultaneously across 6 cores."
echo "==========================================="

# Limit each PyTorch process to 1 thread so we get true core-level parallelism
# (without this, PyTorch tries to use all cores per process and they fight).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Launch in background, each with its own log
declare -a PIDS
for COMBO in "dqn:GBPUSD" "ppo:GBPUSD" "dqn:AUDUSD" "ppo:AUDUSD" "dqn:USDCAD" "ppo:USDCAD"; do
  AGENT="${COMBO%:*}"
  PAIR="${COMBO#*:}"
  RUN_LOG="$LOG_DIR/${AGENT}_${PAIR}_1h.log"
  echo ">>> LAUNCH $AGENT $PAIR 1h  ($(date +%H:%M:%S))  -> $RUN_LOG"
  python demo.py --pair "$PAIR" --agent "$AGENT" --timeframe 1h --seeds 3 \
    > "$RUN_LOG" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All 6 launched. PIDs: ${PIDS[@]}"
echo "Waiting for completion ..."
echo ""

# Wait for each, reporting as they finish
for i in "${!PIDS[@]}"; do
  PID=${PIDS[$i]}
  if wait "$PID"; then
    EXIT=0
  else
    EXIT=$?
  fi
  echo "<<< PID $PID exited with code $EXIT  ($(date +%H:%M:%S))"
done

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))
echo ""
echo "==========================================="
echo "ALL PARALLEL TRAINING COMPLETE — total ${ELAPSED} min"
echo "==========================================="
