#!/bin/bash
# 2x2 grid comparison of the 4 phase4 models on the custom songs.

set -e

DIR_BASELINE="renders/custom_baseline"
DIR_FCS_COM="renders/custom_fcs_com_bilateral"
DIR_FCS1_COM="renders/custom_fcs1_com_bilateral"
DIR_FCS1_BILAT2="renders/custom_fcs1_com_bilat2"
OUT_DIR="renders/custom_comparison"

mkdir -p "$OUT_DIR"

for f in "$DIR_BASELINE"/*.mp4; do
    name=$(basename "$f")
    base="$DIR_BASELINE/$name"
    fcs_com="$DIR_FCS_COM/$name"
    fcs1_com="$DIR_FCS1_COM/$name"
    fcs1_bilat2="$DIR_FCS1_BILAT2/$name"

    if [[ ! -f "$fcs_com" || ! -f "$fcs1_com" || ! -f "$fcs1_bilat2" ]]; then
        echo "[skip] $name — missing in one of the dirs"
        continue
    fi

    out="$OUT_DIR/$name"
    echo "[render] $name"

    ffmpeg -y -loglevel error \
        -i "$base" -i "$fcs_com" -i "$fcs1_com" -i "$fcs1_bilat2" \
        -filter_complex "
            [0:v]pad=iw:ih+50:0:50:color=black,drawtext=text='Baseline (no physics)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v0];
            [1:v]pad=iw:ih+50:0:50:color=black,drawtext=text='FCS w=0.12 + CoM + Bilateral=5 (best FCS)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v1];
            [2:v]pad=iw:ih+50:0:50:color=black,drawtext=text='FCS w=1.0 + CoM + Bilateral=5 (best balance)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v2];
            [3:v]pad=iw:ih+50:0:50:color=black,drawtext=text='FCS w=1.0 + CoM + Bilateral=2 (looser legs)':x=(w-text_w)/2:y=15:fontsize=22:fontcolor=white[v3];
            [v0][v1]hstack=inputs=2[top];
            [v2][v3]hstack=inputs=2[bot];
            [top][bot]vstack=inputs=2[v]
        " \
        -map "[v]" -map 0:a? -c:v libx264 -crf 23 -preset fast \
        "$out"
done

echo ""
echo "Done. Custom-song comparison videos in $OUT_DIR"
ls "$OUT_DIR"
