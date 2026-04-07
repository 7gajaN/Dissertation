#!/bin/bash
# Chain: wait for CoM run, then run foot height, then bilateral
TRAIN_PID=$1

echo "[chain] Watching CoM run (PID $TRAIN_PID)"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[chain] CoM run finished."

echo "[chain] Starting foot height w=0.5..."
/venv/edge/bin/accelerate launch /workspace/Dissertation/train.py \
    --foot_height_loss_weight 0.5 --epochs 500 \
    --exp_name height_w05 --project runs/phase4
echo "[chain] Foot height run finished."

echo "[chain] Starting bilateral w=5.0..."
/venv/edge/bin/accelerate launch /workspace/Dissertation/train.py \
    --bilateral_loss_weight 5.0 --epochs 500 \
    --exp_name bilateral_w5 --project runs/phase4
echo "[chain] Bilateral run finished. All done."
