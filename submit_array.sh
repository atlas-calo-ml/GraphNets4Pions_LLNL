#!/bin/bash
#BSUB -J ml4pion[1-54]
#BSUB -nnodes 1
#BSUB -W 120
#BSUB -G hizphys
#BSUB -o outfiles/regress_crossval_run_%I.out
#BSUB -q pbatch

source ~/.profile.coral

config_file_0="configs/regress_config_$((LSB_JOBINDEX - 1))_fold_0.yaml"
config_file_1="configs/regress_config_$((LSB_JOBINDEX - 1))_fold_1.yaml"
config_file_2="configs/regress_config_$((LSB_JOBINDEX - 1))_fold_2.yaml"
config_file_3="configs/regress_config_$((LSB_JOBINDEX - 1))_fold_3.yaml"

python -u train_block_multiOut_crossval.py --config $config_file_0 &
sleep 2m
python -u train_block_multiOut_crossval.py --config $config_file_1 &
sleep 2m
python -u train_block_multiOut_crossval.py --config $config_file_2 &
sleep 2m
python -u train_block_multiOut_crossval.py --config $config_file_3 &
