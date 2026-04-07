#!/bin/bash
# Wait for training to finish, then eval all three runs
TRAIN_PID=$1

echo "[eval-chain] Watching PID $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[eval-chain] Training finished. Starting eval..."

cd /workspace/Dissertation

for run in fcs_com_combined fcs_w1 fcs_com_bilateral; do
    echo "[eval-chain] Evaluating $run..."
    /venv/edge/bin/python eval_checkpoints.py --run_dir runs/phase4/$run --num_samples 50
    echo "[eval-chain] $run done."
done

echo "[eval-chain] All evaluations complete."
