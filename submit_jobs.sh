#!/bin/sh
# bsub -W 720 -G guests -o 20210603_encodeDecode_model_pion.out ./run_training.sh
bsub -W 720 -G guests -o 20210613_gnBlock_model_pi0.out ./run_training_gnBlock.sh
bsub -W 720 -G guests -o 20210613_gnBlock_model_pi0_concat.out ./run_training_gnBlock_concat.sh
