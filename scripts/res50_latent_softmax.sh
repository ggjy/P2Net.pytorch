#!/usr/bin/env bash

cd ../

GPUID=6
NAME="res50_latent_softmax"
BATCHSIZE=48
FILE_NAME="results"
BLOCK_NUM=1
FUSION='+'


python -u train_duke.py --gpu_ids ${GPUID} --name ${NAME} --batchsize ${BATCHSIZE} --file_name ${FILE_NAME} --block_num ${BLOCK_NUM} --fusion ${FUSION}