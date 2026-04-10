#!/bin/bash
# Render the 3 phase4 models on the custom songs (assumes baseline already done
# and Jukebox features cached in custom_cached/).

set -e
cd /workspace/Dissertation

declare -A MODELS
MODELS[fcs_com_bilateral]="runs/phase4/fcs_com_bilateral/weights/train-2000.pt"
MODELS[fcs1_com_bilateral]="runs/phase4/fcs1_com_bilateral/weights/train-2000.pt"
MODELS[fcs1_com_bilat2]="runs/phase4/fcs1_com_bilat2/weights/train-2000.pt"

for name in fcs_com_bilateral fcs1_com_bilateral fcs1_com_bilat2; do
    ckpt="${MODELS[$name]}"
    echo "[render] $name"
    /venv/edge/bin/accelerate launch test.py \
        --checkpoint "$ckpt" \
        --use_cached_features \
        --feature_cache_dir custom_cached \
        --render_dir "renders/custom_$name" \
        --out_length 30 --use_first_segment \
        > "runs/phase5/custom_${name}.log" 2>&1
    echo "[render] $name done"
done

echo "all 3 models rendered"
