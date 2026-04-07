#!/bin/bash
# Chain: wait for fcs_com_combined, then run higher FCS weight, then full combo
TRAIN_PID=$1

echo "[chain] Watching fcs_com_combined (PID $TRAIN_PID)"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[chain] fcs_com_combined finished."

echo "[chain] Starting FCS w=1.0 (2000 epochs)..."
/venv/edge/bin/accelerate launch /workspace/Dissertation/train.py \
    --fcs_loss_weight 1.0 --fcs_predictor_path models/fcs_predictor.pt \
    --epochs 2000 --exp_name fcs_w1 --project runs/phase4
echo "[chain] FCS w=1.0 finished."

echo "[chain] Starting FCS + CoM + bilateral (2000 epochs)..."
/venv/edge/bin/accelerate launch /workspace/Dissertation/train.py \
    --fcs_loss_weight 0.12 --fcs_predictor_path models/fcs_predictor.pt \
    --com_loss_weight 0.05 --bilateral_loss_weight 5.0 \
    --epochs 2000 --exp_name fcs_com_bilateral --project runs/phase4
echo "[chain] FCS + CoM + bilateral finished. All done."
