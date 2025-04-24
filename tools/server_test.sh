#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
$PYTHON -m torch.distributed.launch --rdzv_endpoint=localhost:29400 --nproc_per_node=$1 main_umil.py -cfg configs/traffic/32_5_server.yaml --output output/umil_test --only_test --pretrained ../weights/best.pth