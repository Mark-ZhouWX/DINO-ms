#!/bin/bash

echo "Run dino in distributed mode on GPU device"

rm -rf device
mkdir device
cp -r ./common ./configs ./model_zoo ./config.py ./eval.py ./train.py ./device
cd ./device

device_ids=4,5,6,7
str_array=(${device_ids//,/ })
device_num=${#str_array[@]}

echo "start training, using device: ${device_ids}, device_num: ${device_num}"
export CUDA_VISIBLE_DEVICES=${device_ids}

mpirun --allow-run-as-root -n ${device_num} python train.py --distributed=True > train.log 2>&1 &

echo "task running in the background, check train.log"
