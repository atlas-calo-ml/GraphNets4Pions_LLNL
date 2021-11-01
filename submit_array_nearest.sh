#!/bin/bash
#BSUB -J ml4pion[1-112]
#BSUB -nnodes 1
#BSUB -W 90
#BSUB -G hizphys
#BSUB -o outfiles/concat_nearest_crossval_run_%I.out
#BSUB -q pbatch

source ~/.profile.coral

config_file_0="configs/nearest_configs/nearest_config_$((LSB_JOBINDEX - 1))_fold_0.yaml"
config_file_1="configs/nearest_configs/nearest_config_$((LSB_JOBINDEX - 1))_fold_1.yaml"
config_file_2="configs/nearest_configs/nearest_config_$((LSB_JOBINDEX - 1))_fold_2.yaml"
config_file_3="configs/nearest_configs/nearest_config_$((LSB_JOBINDEX - 1))_fold_3.yaml"

python -u train_nearest_concat_crossvall.py --config $config_file_0 &
sleep 2m
python -u train_nearest_concat_crossvall.py --config $config_file_1 &
sleep 2m
python -u train_nearest_concat_crossvall.py --config $config_file_2 &
sleep 2m
python -u train_nearest_concat_crossvall.py --config $config_file_3 &
