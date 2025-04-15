#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main.py -cfg configs/msad/32_5_server_train.yaml --batch-size 1 --accumulation-steps 8 --output output/msad --pretrained /home/user01/nikan/UMIL/weights/k400_16_8.pth
