#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 --rdzv_endpoint=localhost:29450 main_attack2.py -cfg configs/traffic/32_5_attack.yaml --batch-size 1 --accumulation-steps 8 --output output/attack2 --pretrained ../weights/best.pth
