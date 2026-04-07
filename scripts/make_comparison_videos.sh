#!/bin/bash
# Combine matching renders from 3 model directories side-by-side with labels.
# Output goes to renders/comparison/<song>.mp4

set -e

DIR_BASELINE="renders/baseline"
DIR_FCS_COM="renders/renders_fcs_com_bilateral"
DIR_FCS1_COM="renders/renders_fcs1_com_bilateral"
OUT_DIR="renders/comparison"

mkdir -p "$OUT_DIR"

# Iterate over the baseline videos and look for matching files in the other two dirs
for f in "$DIR_BASELINE"/*.mp4; do
    name=$(basename "$f")
    base="$DIR_BASELINE/$name"
    fcs_com="$DIR_FCS_COM/$name"
    fcs1_com="$DIR_FCS1_COM/$name"

    if [[ ! -f "$fcs_com" || ! -f "$fcs1_com" ]]; then
        echo "[skip] $name — missing in one of the dirs"
        continue
    fi

    out="$OUT_DIR/$name"
    echo "[render] $name"

    ffmpeg -y -loglevel error \
        -i "$base" -i "$fcs_com" -i "$fcs1_com" \
        -filter_complex "
            [0:v]pad=iw:ih+50:0:50:color=black,drawtext=text='Baseline (no physics)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v0];
            [1:v]pad=iw:ih+50:0:50:color=black,drawtext=text='FCS w=0.12 + CoM + Bilateral (best FCS)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v1];
            [2:v]pad=iw:ih+50:0:50:color=black,drawtext=text='FCS w=1.0 + CoM + Bilateral (best balance)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v2];
            [v0][v1][v2]hstack=inputs=3[v]
        " \
        -map "[v]" -map 0:a? -c:v libx264 -crf 23 -preset fast \
        "$out"
done

echo ""
echo "Done. Comparison videos in $OUT_DIR"
ls "$OUT_DIR"
