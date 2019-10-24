#!/usr/bin/env bash

cd ../

GPUID=5
NAME="res50_softmax"
BATCHSIZE=48
FILE_NAME="results"


python -u train_duke.py --gpu_ids ${GPUID} --name ${NAME} --batchsize ${BATCHSIZE} --file_name ${FILE_NAME}