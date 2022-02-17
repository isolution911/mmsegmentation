#!/usr/bin/env bash

#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-29500}
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

GPUS=4
PORT=${PORT:-29509}

export CUDA_VISIBLE_DEVICES='6,7,8,9'

# phase0 ndwi

# phase1
CONFIG_P1='configs/water/deeplabv3plus_r18-d8_512x512_40k_p1_gidwater.py'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG_P1 --launcher pytorch

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/convert_datasets/generate_pseudo_label.py $CONFIG_P1 \
work_dirs/water/deeplabv3plus_r18-d8_512x512_40k_p1_gidwater/latest.pth \
-i data/GID/img_dir -o data/GID/dlp1t0.9 -t 0.9

# phase2
CONFIG_P2='configs/water/deeplabv3plus_r50-d8_512x512_40k_p2_gidwater.py'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG_P2 --launcher pytorch

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/convert_datasets/generate_pseudo_label.py $CONFIG_P2 \
work_dirs/water/deeplabv3plus_r50-d8_512x512_40k_p2_gidwater/latest.pth \
-i data/GID/img_dir -o data/GID/dlp2t0.7 -t 0.7

# phase3
CONFIG_P3='configs/water/deeplabv3plus_r101-d8_512x512_40k_p3_gidwater.py'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG_P3 --launcher pytorch