#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main.py -cfg configs/traffic/32_5.yaml --batch-size 2 --accumulation-steps 8 --output output/mil --pretrained /kaggle/input/umil-weights/k400_16_8.pth