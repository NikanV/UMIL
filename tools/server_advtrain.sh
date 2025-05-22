#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_advtrain.py -cfg configs/traffic/32_5_advtrain.yaml --batch-size 1 --accumulation-steps 8 --output output/adv_train --pretrained ../weights/best.pth
