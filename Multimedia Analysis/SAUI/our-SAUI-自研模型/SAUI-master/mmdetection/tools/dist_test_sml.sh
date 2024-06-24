#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

$PYTHON -W ignore -m torch.distributed.launch --master_port 10003 --nproc_per_node=$GPUS \
    $(dirname "$0")/test_sml.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
