#!/bin/bash
# Wavelet on the other 4 daily pairs (EUR/USD, USD/JPY, GBP/USD, USD/CAD).
# AUD/USD wavelet was the breakthrough — this round tests if the technique
# helps the already-strong pairs further, stays neutral, or actually hurts.
#
# 4 pairs × 2 agents = 8 parallel detached jobs. 5 seeds × 500k each.
# Estimated wallclock: ~2-3 hours on 8 cores.

cd "$(dirname "$0")"
LOG_DIR="logs/wavelet_full"
mkdir -p "$LOG_DIR"

VENV_PY="$(pwd)/venv/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Launching wavelet trainings for EUR/USD, USD/JPY, GBP/USD, USD/CAD ..."
for PAIR in EURUSD USDJPY GBPUSD USDCAD; do
  for AGENT in dqn ppo; do
    LOG="$LOG_DIR/${AGENT}_${PAIR}_wavelet.log"
    # Double-fork so children survive shell exit
    ( nohup "$VENV_PY" demo.py \
        --pair "$PAIR" --agent "$AGENT" --timeframe 1d \
        --denoise wavelet --seeds 5 \
        > "$LOG" 2>&1 < /dev/null & )
    echo "  launched: $AGENT $PAIR wavelet  -> $LOG"
  done
done

echo ""
echo "8 jobs launched and detached. Track with:"
echo "  ls -la models/*_wavelet.zip"
echo "  tail -f $LOG_DIR/dqn_EURUSD_wavelet.log"
