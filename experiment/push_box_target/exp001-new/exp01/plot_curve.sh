#!/bin/bash

python ../../plot_curve.py --log_file=./train.log --save_display --save_dir=./

convert ./loss.png -trim ./loss.png
convert ./epe.png -trim ./epe.png
