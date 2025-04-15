#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
$PYTHON -m torch.distributed.launch --rdzv_endpoint=localhost:29400 --nproc_per_node=$1 main_umil.py -cfg configs/msad/32_5_server_test.yaml --output output/msad_test --only_test --pretrained /home/user01/nikan/UMIL/UMIL/output/msad/ckpt_epoch_19.pth
