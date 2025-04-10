#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_attack.py -cfg configs/traffic/32_5_attack.yaml --batch-size 1 --accumulation-steps 8 --output output/attack --pretrained /home/user01/nikan/UMIL/weights/best.pth
