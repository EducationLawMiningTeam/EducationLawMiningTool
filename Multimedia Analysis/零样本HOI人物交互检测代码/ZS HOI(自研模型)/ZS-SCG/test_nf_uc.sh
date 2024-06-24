#!/bin/bash


for ((i = 1; i <= 8; i+=1))
do
    echo "Running program with --unseen-model $i"
    CUDA_VISIBLE_DEVICES=2 nohup python test.py  --model-path nf_uc/ckpt_08320_08.pt --unseen-model /data01/liuchuan/zsd2/classifier_latest_199_$((i)).pth --zero-shot-type nf_uc > nf_uc/199_$((i))_real_new.txt 2>&1 & 
done
