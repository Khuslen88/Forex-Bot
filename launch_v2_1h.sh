#!/bin/bash
# Detached launcher for the 1H v2 training wave.
# Uses nohup + disown so the python children survive the parent shell exit.
# Logs go to logs/v2_training/, models save to models/.
#
# After launching, this script exits immediately. Track progress via:
#   ls -la models/*_1h_v2.zip                    # model files appear when each finishes
#   tail -f logs/v2_training/dqn_EURUSD_1h_v2.log

cd "$(dirname "$0")"

LOG_DIR="logs/v2_training"
mkdir -p "$LOG_DIR"

VENV_PY="$(pwd)/venv/bin/python"

# 1 thread per process for true core-level parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Launching 10 detached 1H v2 trainings ..."
for PAIR in EURUSD USDJPY GBPUSD AUDUSD USDCAD; do
  for AGENT in dqn ppo; do
    LOG="$LOG_DIR/${AGENT}_${PAIR}_1h_v2.log"
    # Double-fork via subshell so the python is reparented to init/launchd
    # and survives this script's exit. nohup ignores SIGHUP for extra safety.
    ( nohup "$VENV_PY" demo.py \
        --pair "$PAIR" --agent "$AGENT" --timeframe 1h \
        --version v2 --seeds 1 \
        > "$LOG" 2>&1 < /dev/null & )
    echo "  launched: $AGENT $PAIR 1h v2  -> $LOG"
  done
done

echo ""
echo "All 10 launched and detached. They will continue running after this script exits."
echo "Models will appear in models/*_1h_v2.zip as each one finishes (~60-90 min each)."
