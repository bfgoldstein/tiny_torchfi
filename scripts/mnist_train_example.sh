#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH=$PYTHONPATH:/Users/bfgoldstein/workspace/tiny_torchfi
export PYTHONPATH=$PYTHONPATH:/home/bruno/tiny_torchfi

# Train a full-precision baseline with golden and faulty models
python ../examples/mnist_train.py -a mnist -j 16 --batch-size=128 --test-batch-size=128 --weight-decay=0  --gamma=0  --learning-rate=0.01 --momentum=0.5 --epochs 20 --golden --faulty --injection --layer=1 --weights --fiEpoch=1 /Users/bfgoldstein/workspace/dataset
