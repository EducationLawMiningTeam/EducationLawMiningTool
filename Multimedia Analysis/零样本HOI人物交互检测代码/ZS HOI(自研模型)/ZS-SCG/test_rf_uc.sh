#!/bin/bash
for ((i = 7; i <= 12; i+=1))
do
    echo "Running program with --unseen-model $i"
    CUDA_VISIBLE_DEVICES=4 nohup python test.py  --model-path rf_uc/ckpt_10948_07.pt --unseen-model /data01/liuchuan/zsd2/results_rf_uc_all_obj_loss/classifier_latest_249_$((i)).pth --zero-shot-type rf_uc > rf_uc/249_$((i)).txt 2>&1 & 
done
