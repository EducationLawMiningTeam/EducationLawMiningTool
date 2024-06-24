#!/bin/bash


for ((i = 1; i <= 9; i+=1))
do
    echo "Running program with --unseen-model $i"
    CUDA_VISIBLE_DEVICES=2 nohup python test.py  --model-path ua/ckpt_07945_07.pt --unseen-model /data01/liuchuan/zsd2/results_ua_con/classifier_latest_269_$((i)).pth --zero-shot-type ua > ua/con_269_$((i)).txt 2>&1 & 
done
