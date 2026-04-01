#!/bin/bash
# Launch FCS training after 3-hour delay
# Started at: $(date)

LOG=/workspace/Dissertation/scripts/fcs_training.log

echo "[$(date)] Waiting 3 hours for baseline training to finish..." >> $LOG

sleep 10800  # 3 hours

echo "[$(date)] Starting FCS training..." >> $LOG

# Check if baseline training is still running
if pgrep -f "train.py.*runs/baseline" > /dev/null 2>&1; then
    echo "[$(date)] WARNING: Baseline training still running. Waiting for it to finish..." >> $LOG
    while pgrep -f "train.py.*runs/baseline" > /dev/null 2>&1; do
        sleep 300  # check every 5 minutes
    done
    echo "[$(date)] Baseline training finished. Proceeding." >> $LOG
fi

cd /workspace/Dissertation
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate edge2

accelerate launch train.py \
    --project runs/fcs \
    --exp_name physics_v1 \
    --batch_size 128 \
    --epochs 2000 \
    --save_interval 100 \
    --feature_type jukebox \
    --fcs_predictor_path models/fcs_predictor.pt \
    --fcs_loss_weight 0.1 \
    >> $LOG 2>&1

echo "[$(date)] FCS training completed." >> $LOG
