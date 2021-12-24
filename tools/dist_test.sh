#!/usr/bin/env bash

#CONFIG=$1
#CHECKPOINT=$2
#GPUS=$3
#PORT=${PORT:-29500}
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

CONFIG=$1
CHECKPOINT=$2
GPUS=4
PORT=${PORT:-29502}

export CUDA_VISIBLE_DEVICES='2,3,4,5'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:3}
