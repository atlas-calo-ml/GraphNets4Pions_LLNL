#!/bin/sh
# bsub -W 720 -G guests -o 20210523_encodeDecode_model.out ./run_training.sh
bsub -W 720 -G hizphys -o 20210601_lassen_gnBlock_model.out ./run_training_gnBlock.sh
bsub -W 720 -G hizphys -o 20210601_lassen_gnBlock_model_concatTrue.out ./run_training_gnBlock_concat.sh
