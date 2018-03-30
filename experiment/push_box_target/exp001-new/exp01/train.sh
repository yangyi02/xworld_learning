#!/bin/bash

source ../../../../set_env.sh

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python3 ../main_a3c.py --train --tensorboard_path=../../tensorboard/exp001_01 --map_config=../empty_ground.json --save_dir=./ --learning_rate=0.0001 2>&1 | tee train.log
