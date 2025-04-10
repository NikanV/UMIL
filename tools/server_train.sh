#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main.py -cfg configs/traffic/32_5_server.yaml --batch-size 1 --accumulation-steps 8 --output output/mil2 --pretrained /home/user01/nikan/UMIL/weights/k400_16_8.pth
