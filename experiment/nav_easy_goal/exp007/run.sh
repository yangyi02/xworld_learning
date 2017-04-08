#!/bin/bash

# usage: ./run.sh <testing> <sampler-type> <gpu-id>

TEST=0
(($# > 0)) && TEST=$1
SAMPLER="RankBased"
(($# > 1)) && SAMPLER=$2
GPU=1
(($# > 2)) && GPU=$3

if [ "$TEST" -eq "0" ]; then
    # training
    SHOW_FRAME=0
    NUM_GAMES=500000  # for training
    INIT_MODEL=""
    
    # Create directories if not exist
    if [[ ! -e ./models ]] && [[ ! -e ./log ]]; then
        mkdir -p ./models
        mkdir -p ./log
    else
        echo "./models or ./log already exist!"
        echo "If you really want to run the experiment with same name,"
        echo "rm -rf ./models ./log"
        exit
    fi
    
    OMP_NUM_THREADS=1 python3 main_a3c.py --train 2>&1 | tee ./log/train.log
else
    # testing
    SHOW_FRAME=1
    NUM_GAMES=500    # for benchmarking
    INIT_MODEL="./models/final.pth"
    
    if [ "$SHOW_FRAME" -eq "1" ]; then
        python main_a3c.py --test --init_model=$INIT_MODEL --show_frame 2>&1 | tee ./log/test.log
    else
        python main_a3c.py --test --init_model=$INIT_MODEL 2>&1 | tee ./log/test.log
    fi
        
fi

