#!/bin/bash
set -e

cd "$(dirname "$0")/.."

MODEL=/home/gao/codes/TeleScenegraphFusion-visual-feature/out/telescenegraphfusion_sgfn_yolo_label_relation
DATA_ROOT=/media/gao/gao_ssd/teledataset
SG_ROOT="/media/gao/gao_ssd/3D scene graph"

for i in $(seq 1 84); do
    for j in 1 2 3; do
        SEQ="${i}_${j}"
        PTH_IN="${DATA_ROOT}/${SEQ}"
        PTH_OUT="${SG_ROOT}/${SEQ}"

        if [ ! -d "$PTH_IN" ]; then
            echo "[SKIP] ${SEQ}: dataset not found"
            continue
        fi

        echo "[START] ${SEQ} -> ${PTH_OUT}"
        ./bin/exe_GraphSLAM \
            --pth_in    "$PTH_IN" \
            --pth_model "$MODEL"  \
            --scene_graph_json_dir "$PTH_OUT"
        echo "[DONE]  ${SEQ}"
        sleep 10
        echo "---"
    done
done

echo "All sequences finished."
