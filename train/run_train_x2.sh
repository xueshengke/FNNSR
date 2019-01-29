#!/usr/bin/env sh
set -e

LOG=examples/FNNSR/log/train_FNNSR_x2_`date +%Y-%m-%d-%H-%M-%S`.log

CAFFE=./build/tools/caffe

# run this .sh file in the Caffe root directory

$CAFFE train --solver=examples/FNNSR/solver_x2.prototxt $@ 2>&1 | tee $LOG

# args
# --solver --gpu -- snapshot --weights --iterations --model

