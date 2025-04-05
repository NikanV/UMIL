#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_umil.py -cfg configs/traffic/32_5.yaml --output output/test --only_test --pretrained /kaggle/input/umil-weights/k400_16_8.pth
