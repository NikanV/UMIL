#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_umil.py -cfg configs/traffic/32_5_server.yaml --output output/test --only_test --pretrained /home/user01/nikan/UMIL/weights/best.pth
