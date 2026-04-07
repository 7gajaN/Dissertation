#!/bin/bash
# Wait for training process to finish, then run evaluation
TRAIN_PID=$1
RUN_DIR=$2
NUM_SAMPLES=${3:-50}

echo "[eval-scheduler] Watching PID $TRAIN_PID, will evaluate $RUN_DIR when done"

# Wait for the training process to exit
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done

echo "[eval-scheduler] Training finished (PID $TRAIN_PID exited). Starting evaluation..."
cd /workspace/Dissertation
source activate edge 2>/dev/null || conda activate edge 2>/dev/null

python eval_checkpoints.py --run_dir "$RUN_DIR" --num_samples "$NUM_SAMPLES"
echo "[eval-scheduler] Evaluation complete. Results in $RUN_DIR/eval_results.json"
