#!/bin/bash
# Versuche Conda-Pfad automatisch zu finden (Miniconda oder Anaconda)
# Passt für Standard-Installationen auf Ubuntu
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    # Fallback: Versuche es über den Pfad, falls conda global verfügbar ist
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        echo "FEHLER: Conda installation nicht gefunden."
        exit 1
    fi
fi

# Removing previous outputs/home/nadezhda/RGMC/realsense/out1/masks/1.png
echo "This will remove previous outputs in /home/nadezhda/RGMC/realsense/out1/"

read -p "Remove previous outputs? [y/N]: " confirm

if [[ ! "$confirm" =~ ^[yY]$ ]]; then
    exit 0
fi

rm /home/nadezhda/RGMC/realsense/out1/depth/*
rm /home/nadezhda/RGMC/realsense/out1/rgb/*
rm /home/nadezhda/RGMC/realsense/out1/masks/*

echo "Removal complete."

# Realsense - Depth and RGB Stream 

cd /home/nadezhda/RGMC || exit 1

cd /home/nadezhda/RGMC/realsense

source ./venv/bin/activate

python ./export_aligned_streams.py &
PID=$!

sleep 5

kill $PID

wait $PID 2>/dev/null

deactivate


# YOLO - Object Detection

cd /home/nadezhda/RGMC/yolo_temp

source ./venv/bin/activate

python test_yolo.py --input /home/nadezhda/RGMC/realsense/out1/rgb  --no-show

deactivate

# SAM3

cd /home/nadezhda/RGMC/

conda activate sam3

cd sam3

python test_bbox_prompt_from_yolo.py --input_img /home/nadezhda/RGMC/realsense/out1/rgb --yolo_results /home/nadezhda/RGMC/yolo_temp/output/results.txt --output /home/nadezhda/RGMC/realsense/out1/masks --mode top

conda deactivate

# Dimension Calculation
cd /home/nadezhda/RGMC/yolo_temp

source ./venv/bin/activate

cd /home/nadezhda/RGMC/object_dimensions_temp

python calculate_object_dim.py \
    --yolo_results /home/nadezhda/RGMC/yolo_temp/output/results.txt \
    --depth_dir /home/nadezhda/RGMC/realsense/out1/depth \
    --mask_dir /home/nadezhda/RGMC/realsense/out1/masks \
    --intrinsics /home/nadezhda/RGMC/realsense/realsense_intrinsics.yml

deactivate
