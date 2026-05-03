#!/bin/bash
# Round 1: 5 seeds x 1M timesteps for the 3 weak pairs (EURUSD, AUDUSD, GBPUSD)
# x both agents (DQN + PPO) = 6 parallel training jobs.
#
# Each job sequentially trains 5 seeds and keeps the best by val Sharpe.
# Models save to the standard v1 location, OVERWRITING current weak v1 models.
# (We backed up models/backup_round1/ first — restore from there if results regress.)
#
# Estimated wallclock: ~3-5 hours on 8 cores, ~700 MB RAM each.

cd "$(dirname "$0")"
LOG_DIR="logs/round1_training"
mkdir -p "$LOG_DIR"

VENV_PY="$(pwd)/venv/bin/python"

# 1 thread per process for true core-level parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Launching 6 detached Round 1 trainings ..."
for PAIR in EURUSD AUDUSD GBPUSD; do
  for AGENT in dqn ppo; do
    LOG="$LOG_DIR/${AGENT}_${PAIR}_round1.log"
    # Double-fork so children survive shell exit
    ( nohup "$VENV_PY" demo.py \
        --pair "$PAIR" --agent "$AGENT" --timeframe 1d \
        --seeds 5 \
        > "$LOG" 2>&1 < /dev/null & )
    echo "  launched: $AGENT $PAIR  -> $LOG"
  done
done

echo ""
echo "All 6 launched and detached."
echo ""
echo "Track progress:"
echo "  ls -la models/dqn_feature_*.zip models/ppo_feature_*.zip"
echo "  tail -f logs/round1_training/dqn_EURUSD_round1.log"
echo ""
echo "When all complete, run:"
echo "  python src/utils/build_registry.py"
