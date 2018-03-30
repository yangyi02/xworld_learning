#!/bin/bash

source ../../set_path.sh

python3 ../main_a3c.py --test --init_model=./model.pth --show_frame 2>&1 | tee test.log
