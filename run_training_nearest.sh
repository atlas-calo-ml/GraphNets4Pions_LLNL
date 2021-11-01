#!/bin/sh

source ~/.profile.coral

# python train_block_multiOut.py --config configs/simult_optimized.yaml &
# sleep 2m
python train_nearest_weightedRegress.py --config configs/nearest_weightedRegress_optimized.yaml &
sleep 1m
python train_nearest_concat.py --config configs/nearest_concatClass_optimized.yaml

