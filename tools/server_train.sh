#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --rdzv_endpoint=localhost:29450 --nproc_per_node=$1 main.py -cfg configs/traffic/32_5_server.yaml --batch-size 1 --accumulation-steps 8 --output output/mil --pretrained ../weights/k400_16_8.pth
