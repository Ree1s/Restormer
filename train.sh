#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
CONFIG=$1

python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch