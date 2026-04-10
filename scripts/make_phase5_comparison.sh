#!/bin/bash
# Phase 5 qualitative comparison: baseline λ=0 vs baseline λ=1 vs Phase 4 best.
# Three columns side-by-side with labels, output to renders/phase5_comparison/.

set -e

DIR_BASE_L0="renders/phase5_baseline_l0"
DIR_BASE_L1="renders/phase5_baseline_l1"
DIR_PHASE4="renders/renders_fcs_com_bilateral"
OUT_DIR="renders/phase5_comparison"

mkdir -p "$OUT_DIR"

for f in "$DIR_BASE_L0"/*.mp4; do
    name=$(basename "$f")
    base_l0="$DIR_BASE_L0/$name"
    base_l1="$DIR_BASE_L1/$name"
    phase4="$DIR_PHASE4/$name"

    if [[ ! -f "$base_l1" || ! -f "$phase4" ]]; then
        echo "[skip] $name — missing in one of the dirs"
        continue
    fi

    out="$OUT_DIR/$name"
    echo "[render] $name"

    ffmpeg -y -loglevel error \
        -i "$base_l0" -i "$base_l1" -i "$phase4" \
        -filter_complex "
            [0:v]pad=iw:ih+50:0:50:color=black,drawtext=text='Baseline (no physics, no guidance)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v0];
            [1:v]pad=iw:ih+50:0:50:color=black,drawtext=text='Baseline + inference guidance (λ=1.0)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v1];
            [2:v]pad=iw:ih+50:0:50:color=black,drawtext=text='Phase 4 best (training-time physics)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v2];
            [v0][v1][v2]hstack=inputs=3[v]
        " \
        -map "[v]" -map 0:a? -c:v libx264 -crf 23 -preset fast \
        "$out"
done

echo ""
echo "Done. Phase 5 comparison videos in $OUT_DIR"
ls "$OUT_DIR"
