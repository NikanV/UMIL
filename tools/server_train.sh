#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_umil.py -cfg configs/traffic/32_5_server.yaml --batch-size 1 --batch-size-umil 1 --accumulation-steps 8 --output output/umil --pretrained ../weights/k400_16_8.pth
