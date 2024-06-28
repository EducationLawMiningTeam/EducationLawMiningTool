# coding=UTF-8 

import os
import datetime
import random
import time
import csv
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score
import argparse
import numpy as np
from model import *
from utils import *
import copy
from speed import *

threshold=0.5      
early_stopp_patience=3
detach = lambda o: o.cpu().detach().numpy().tolist()

def set_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_speed(a):
    for key in speed_dict:
        if a<=float(key):
            return int(speed_dict[key])

class Data:
    def __init__(self, file, length, q_num,speed_cate, is_test=False, index_split=None, is_train=False):
        rows = csv.reader(file, delimiter=',')
        rows = [[int(e) for e in row if e != ''] for row in rows]
        q_rows, r_rows, skill_rows,spend_rows,last_rows,repeat_rows,done_rows,fast_rows = [], [], [],[], [], [],[],[]
        student_num = 0
        global speed_dict
        if speed_cate==2:
            speed_dict=speed_dict_2
        if speed_cate==5:
            speed_dict=speed_dict_5
        if speed_cate==10:
            speed_dict=speed_dict_10
        if speed_cate==19:
            speed_dict=speed_dict_19

        if is_test:
            for q_row, r_row,skill_row,spend_row,fast_row,last_row,repeat_row,done_row in zip(rows[1::10], rows[2::10],rows[3::10],rows[4::10],rows[5::10],rows[7::10],rows[8::10],rows[9::10]):
                q_rows.append(q_row[0:length])
                r_rows.append(r_row[0:length])
                skill_rows.append(skill_row[0:length])

                fast_row=np.array(fast_row)
                fast_row[fast_row==0]=1
                spend_rows.append(list(map(get_speed,(np.array(spend_row)/fast_row).tolist()))[0:length])
                last_row=np.array(last_row)
                last_row[last_row==0]=1036800
                last_row=last_row.tolist()
                last_rows.append(last_row[0:length])
                repeat_rows.append(repeat_row[0:length])
                done_rows.append(done_row[0:length])         

        else:
            if is_train:
                for q_row, r_row,skill_row,spend_row,fast_row,last_row,repeat_row,done_row in zip(rows[1::10], rows[2::10],rows[3::10],rows[4::10],rows[5::10],rows[7::10],rows[8::10],rows[9::10]):
                    if student_num not in index_split:              
                        q_rows.append(q_row[0:length])
                        r_rows.append(r_row[0:length])
                        skill_rows.append(skill_row[0:length])

                        fast_row=np.array(fast_row)
                        fast_row[fast_row==0]=1
                        spend_rows.append(list(map(get_speed,(np.array(spend_row)/fast_row).tolist()))[0:length])
                        last_row=np.array(last_row)
                        last_row[last_row==0]=1036800
                        last_row=last_row.tolist()
                        last_rows.append(last_row[0:length])
                        repeat_rows.append(repeat_row[0:length])
                        done_rows.append(done_row[0:length])      
                    student_num += 1

            else:
                for q_row, r_row,skill_row,spend_row,fast_row,last_row,repeat_row,done_row in zip(rows[1::10], rows[2::10],rows[3::10],rows[4::10],rows[5::10],rows[7::10],rows[8::10],rows[9::10]):
                    if student_num in index_split:
                        q_rows.append(q_row[0:length])
                        r_rows.append(r_row[0:length])
                        skill_rows.append(skill_row[0:length])

                        fast_row=np.array(fast_row)
                        fast_row[fast_row==0]=1
                        spend_rows.append(list(map(get_speed,(np.array(spend_row)/fast_row).tolist()))[0:length])
                        last_row=np.array(last_row)
                        last_row[last_row==0]=1036800
                        last_row=last_row.tolist()
                        last_rows.append(last_row[0:length])
                        repeat_rows.append(repeat_row[0:length])
                        done_rows.append(done_row[0:length])  
                    student_num += 1
        done_rows = [np.round((row-np.ones(len(row))*row[0])/1000000).astype(int) for row in done_rows]
        self.q_rows = q_rows
        self.r_rows = r_rows
        self.spend_rows = spend_rows 
        self.skill_rows = skill_rows 
        self.last_rows = last_rows 
        self.repeat_rows = repeat_rows 
        self.done_rows = done_rows 
        self.q_num = q_num

    def __getitem__(self, index):
        return list(
            zip(self.q_rows[index], self.r_rows[index],self.skill_rows[index],
            self.spend_rows[index],
            self.last_rows[index],self.repeat_rows[index],self.done_rows[index]
            ))
    def __len__(self):
        return len(self.q_rows)

def collate(batch,length):
    batch =torch.tensor([[[*e, 1] for e in row] + [[0, 0, 0,0,0,0,0,0]] * (length - len(row)) for row in batch]).cuda()   #修改
    Q, Y, Skill,Spend, Last,Repeat, Done,S = batch.T 
    Q, Y, Skill,Spend, Last,Repeat, Done,S =Q.T, Y.T, Skill.T,Spend.T, Last.T,Repeat.T, Done.T,S.T 
    return Q, Y, Skill,Spend, Last,Repeat, Done,S

def all_loss(bce_loss,cross_ent_loss):
    return (1-Loss_rate)*bce_loss+cross_ent_loss*Loss_rate

def train(model, data, optimizer, batch_size,length):
    model.train(mode=True)
    y_criterion = nn.BCELoss()
    time_criterion=nn.CrossEntropyLoss()
    for Q, Y, Skill,Spend, Last,Repeat, Done,S in DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=lambda batch: collate(batch,length),
            shuffle=True
    ):
        P,Y,S,out_time,Spend,out2 = model(Q, Y, Skill,Spend, Last,Repeat, Done,S)
        index = S == 1
        y_loss = y_criterion(P[index], Y[index].float())
        time_loss=time_criterion(out_time[index], Spend[index])
        loss_all=all_loss(bce_loss=y_loss,cross_ent_loss=time_loss)
        optimizer.zero_grad()
        loss_all.backward()
        print("ALL_LogisticLoss: %.4f " % loss_all)

        optimizer.step()

def evaluate(model, data, batch_size,length):
    model.eval()
    y_criterion = nn.BCELoss()
    time_criterion=nn.CrossEntropyLoss()
    y_pred, y_true = [], []
    y_loss = 0.0
    time_loss=0.0
    for Q, Y, Skill,Spend, Last,Repeat, Done,S in DataLoader(
            dataset=data,
            batch_size=batch_size,
            collate_fn=lambda batch: collate(batch,length)
    ):
        P,Y,S,out_time,Spend,out2 = model(Q, Y, Skill,Spend, Last,Repeat, Done,S)
        index = S == 1
        P, Y = P[index], Y[index].float()
        out_time,Spend=out_time[index], Spend[index]
        y_pred += detach(P)
        y_true += detach(Y)
        y_loss+= detach(y_criterion(P, Y) * P.shape[0])
        time_loss += detach(time_criterion(out_time, Spend) * P.shape[0])
    loss_all=all_loss(bce_loss=y_loss,cross_ent_loss=time_loss)
    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    mse_value = mean_squared_error(y_true, y_pred)
    mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= threshold else 0 for i in y_pred]
    acc_value = accuracy_score(y_true, bi_y_pred)
    return auc(fpr, tpr), loss_all / len(y_true), mse_value, mae_value, acc_value

def experiment(dataset, hidden_num,learning_rate,epochs,batch_size,seed,cv_num,q_num,length,
        time_spend,d_model,nhead,num_encoder_layers,dropout,gpu,speed_cate,loss_rate):
    global Loss_rate
    Loss_rate=loss_rate
    if gpu=='0':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if gpu=='1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    set_seed(seed)
    path = 'results/'+str(dataset)+'_exp/%s' % ('{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()))
    os.makedirs(path)
    info_file = open('%s/info.txt' % path, 'w+')
    info_file1 = open('%s/info1.txt' % path, 'w+')
    save_info_0(dataset, hidden_num,learning_rate,epochs,batch_size,seed,cv_num,q_num,length,
        time_spend,d_model,nhead,num_encoder_layers,dropout,gpu,speed_cate,loss_rate,info_file)

    test_data = Data(open('data/%s/test.csv' % dataset, 'r'), length, q_num, is_test=True,speed_cate=speed_cate)
    valid_list,auc_list,mse_list,mae_list,acc_list,loss_list = [],[],[],[],[],[]

    '''训练过程'''
    for cv in range(cv_num):
        if dataset=='junyi_all':
            origin_list = [i for i in range(172589)]
        if dataset=='junyi_for_testing':
            origin_list = [i for i in range(400)]
        
        random.seed(cv + 1000)
        index_split = random.sample(origin_list, int(0.05 * len(origin_list)))
        random.seed(0)

        train_data = Data(open('data/%s/train_valid.csv' % dataset, 'r'), length, q_num, is_test=False,
                          index_split=index_split, is_train=True,speed_cate=speed_cate)
        valid_data = Data(open('data/%s/train_valid.csv' % dataset, 'r'), length, q_num, is_test=False,
                          index_split=index_split, is_train=False,speed_cate=speed_cate)

        if gpu in ['one','0','1']:
            model = Model_exp(q_num, time_spend,d_model, length,nhead,num_encoder_layers, dropout,speed_cate).cuda()
        if gpu=='two':
            model = Model_exp(q_num, time_spend,d_model, length,nhead,num_encoder_layers, dropout,speed_cate).cuda()
            torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
            model = nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=early_stopp_patience, verbose=True)

        test_max_auc=0.0
        max_auc = 0.0
        for epoch in range(1, epochs + 1):
            time_start = time.time()
            train(model, train_data, optimizer, batch_size,length)
            train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model, train_data, batch_size,length)
            valid_auc, valid_loss, valid_mse, valid_mae, valid_acc = evaluate(model, valid_data, batch_size,length)
            # test_auc, test_loss, test_mse, test_mae, test_acc = evaluate(model, test_data, batch_size,length)
            time_end = time.time()
            if max_auc < valid_auc:
                max_auc = valid_auc
                torch.save(model.state_dict(), '%s/model_%s' % ('%s' % path, '%d' % cv))
                current_max_model = copy.deepcopy(model)
            save_info_1(cv,epoch,max_auc,valid_auc,valid_loss,valid_mse,valid_mae,valid_acc,train_auc,train_loss, 
                        train_mse,train_mae,train_acc,time_end,time_start,info_file,info_file1)
            # save_info_4(cv,valid_auc, test_auc, test_mse, test_mae,test_acc,test_loss,info_file)
            #
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        valid_auc, valid_loss, valid_mse, valid_mae, valid_acc = evaluate(current_max_model, valid_data, batch_size,length)
        test_auc, test_loss, test_mse, test_mae, test_acc = evaluate(current_max_model, test_data, batch_size,length)
        save_info_3(valid_auc, test_auc,test_mse, test_mae, test_acc,test_loss,info_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to test DKT.')
    parser.add_argument('--dataset', type=str, default='junyi_for_testing', help='')
    parser.add_argument('--hidden_num', type=int, default=512, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cv_num', type=int, default=1, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--q_num', type=int, default=720, help='')
    parser.add_argument('--length', type=int, default=300, help='')
    parser.add_argument('--time_spend', type=int, default=172, help='')
    parser.add_argument('--d_model', type=int, default=512, help='')
    parser.add_argument('--nhead', type=int, default=8, help='')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='')
    parser.add_argument('--dropout', type=float, default=0, help='')
    parser.add_argument('--gpu', type=str, default='1', help='')
    parser.add_argument('--speed_cate', type=int, default='10', help='')
    parser.add_argument('--loss_rate', type=float, default='0.25', help='')
    params = parser.parse_args()
    dataset = params.dataset

    if dataset == 'junyi_for_testing':
        params.q_num = 720
        params.length = 200
    if dataset == 'junyi_all':
        params.q_num = 720
        params.length = 200

    experiment(
        dataset = params.dataset,
        hidden_num = params.hidden_num,
        learning_rate = params.learning_rate,
        epochs = params.epochs,
        batch_size = params.batch_size,
        seed = params.seed,
        cv_num=params.cv_num,
        q_num=params.q_num ,
        length=params.length,
        time_spend=params.time_spend,
        d_model=params.d_model,
        nhead=params.nhead,
        num_encoder_layers=params.num_encoder_layers,
        dropout=params.dropout,
        gpu=params.gpu,
        speed_cate=params.speed_cate,
        loss_rate=params.loss_rate
    )

