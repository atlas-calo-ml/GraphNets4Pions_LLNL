#!/bin/bash
#BSUB -J ml4pion
#BSUB -nnodes 1
#BSUB -W 720
#BSUB -G hizphys
#BSUB -o output/big_run.out
#BSUB -e output/big_run.err
#BSUB -q pbatch

source /usr/workspace/pierfied/opence/bin/activate

python -u train_block.py --config configs/big_run.yaml
