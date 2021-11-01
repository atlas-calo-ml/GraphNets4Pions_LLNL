#!/bin/sh

python save_nearest_data.py --config configs/nearest_weightedRegress_optimized.yaml --k 4
python save_nearest_data.py --config configs/nearest_weightedRegress_optimized.yaml --k 6
python save_nearest_data.py --config configs/nearest_weightedRegress_optimized.yaml --k 8
