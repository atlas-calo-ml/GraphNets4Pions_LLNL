#!/bin/sh

source ~/.profile.coral

# python train_block_multiOut.py --config configs/simult_optimized.yaml &
# sleep 2m
# python train_multiOut_weightedRegress.py --config configs/simult_weightedRegress_optimized.yaml &
# sleep 2m
python train_multiOut_concatClass.py --config configs/simult_concatClass_optimized.yaml
