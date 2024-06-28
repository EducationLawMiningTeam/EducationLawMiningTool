# coding: utf-8
# 2023/11/21 @ xubihan
import numpy as np
from load_data import DATA

import argparse

import sys
import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from LKTcell import LKT
import logging


def generate_q_matrix(path, n_skill, n_problem, gamma=0):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        q_matrix[p][problem2skill[p]] = 1
    return q_matrix

# LBKT參數
n_skills = 123
memory_size = n_skills + 1
n_exercises = 17751

seqlen = 100
dim_tp = 128
num_resps = 2
num_units = 128
dropout = 0.2
dim_hidden = 50
batch_size = 8
q_gamma = 0.1

dat = DATA(seqlen=seqlen, separate_char=',')
data_path = '../../data/2009_skill_builder_data_corrected/LBKT_d/'
train_data = dat.load_data(data_path + 'train.txt')
test_data = dat.load_data(data_path + 'test.txt')
# 17752 * 124
q_matrix = generate_q_matrix(
    data_path + 'problem2skill',
    n_skills, n_exercises,
    q_gamma
)

# FKT參數
q_num = 17751
time_spend = 172 # ?
d_model = 128 # embedding维度 = dim_tp
length = 100
nhead = 8
num_encoder_layers = 6
# dropout = 0
speed_cate = 10 # ?

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to test LKT.')

    parser.add_argument('--seed', type=int, default=0, help='')

  
    params = parser.parse_args()

    lkt = LKT(q_num, n_exercises, time_spend, d_model, dim_tp, length, nhead, num_encoder_layers, speed_cate, num_resps, num_units, dropout,
              dim_hidden, memory_size, batch_size, q_matrix, params.seed)
    
    # lbkt = LKT(n_exercises, dim_tp, num_resps, num_units, dropout,
    #             dim_hidden, memory_size, batch_size, q_matrix)
    
    lkt.train(train_data, test_data, epoch=2, lr=0.001)
    lkt.save("lbkt.params")
    
    lkt.load("lbkt.params")
    _, auc, accuracy, rmse = lkt.eval(test_data)
    print("auc: %.6f, accuracy: %.6f, rmse: %.6f" % (auc, accuracy, rmse))
