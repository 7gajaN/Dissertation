#!/bin/bash
# Hourly monitor for FCS training — checks if FCS loss is active and working
LOG=/workspace/Dissertation/scripts/fcs_monitor.log
METRICS_DIR="/workspace/Dissertation/runs/fcs"

echo "" >> $LOG
echo "========================================" >> $LOG
echo "[$(date)] Hourly FCS Training Check" >> $LOG
echo "========================================" >> $LOG

# Check if training is running
if pgrep -f "train.py.*runs/fcs" > /dev/null 2>&1; then
    echo "Status: RUNNING" >> $LOG
else
    # Check if it hasn't started yet
    if pgrep -f "launch_fcs_training" > /dev/null 2>&1; then
        echo "Status: WAITING (launch script running, training not started yet)" >> $LOG
        echo "========================================" >> $LOG
        exit 0
    else
        echo "Status: NOT RUNNING" >> $LOG
        echo "========================================" >> $LOG
        exit 0
    fi
fi

# Find the metrics CSV
METRICS_CSV=$(find "$METRICS_DIR" -name "training_metrics.csv" 2>/dev/null | head -1)

if [ -z "$METRICS_CSV" ]; then
    echo "No metrics file found yet" >> $LOG
    echo "========================================" >> $LOG
    exit 0
fi

echo "Metrics file: $METRICS_CSV" >> $LOG

# Show latest entries
echo "" >> $LOG
echo "Latest metrics:" >> $LOG
tail -5 "$METRICS_CSV" >> $LOG

# Check FCS loss column (8th column) in checkpoint rows
echo "" >> $LOG
echo "FCS Loss values from checkpoints:" >> $LOG
grep "checkpoint" "$METRICS_CSV" | awk -F',' '{printf "  Epoch %s: FCS_Loss=%s, FCS_Score=%s, PFC_Score=%s\n", $1, $8, $9, $10}' >> $LOG

# Validate FCS loss is non-zero (column 8)
FCS_VALUES=$(grep "checkpoint" "$METRICS_CSV" | awk -F',' '{print $8}' | grep -v "N/A" | grep -v "0.000000")
if [ -z "$FCS_VALUES" ]; then
    echo "" >> $LOG
    echo "WARNING: FCS loss appears to be zero or N/A in all checkpoints!" >> $LOG
    echo "The FCS predictor may not be working correctly." >> $LOG
else
    echo "" >> $LOG
    echo "OK: FCS loss is active and non-zero" >> $LOG
fi

# GPU utilization
echo "" >> $LOG
echo "GPU Status:" >> $LOG
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader >> $LOG 2>&1

echo "========================================" >> $LOG
