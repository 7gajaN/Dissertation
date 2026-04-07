#!/bin/bash
# Wait for a training process to finish, then launch the next one.
# Usage: nohup bash scripts/next_training.sh <PID> -- <train args...> > log.txt 2>&1 &
TRAIN_PID=$1
shift; shift  # remove PID and --

echo "[scheduler] Watching PID $TRAIN_PID, will launch next training when done"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done

echo "[scheduler] Training finished. Launching next run..."
cd /workspace/Dissertation
/venv/edge/bin/accelerate launch train.py "$@"
