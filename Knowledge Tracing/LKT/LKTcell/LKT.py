# coding: utf-8
# 2023/11/21 @ xubihan

from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_curve, auc, mean_absolute_error, accuracy_score
import logging
import torch
import torch.nn as nn
import numpy as np
import random
import os
# FKT
from .model import Model_exp
# LBKL
from .model import Recurrent

from EduKTM import KTM
from tqdm import tqdm
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) \
        + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0

def all_loss(bce_loss,cross_ent_loss):
    Loss_rate = 0.25
    return (1-Loss_rate)*bce_loss+cross_ent_loss*Loss_rate

# Topics_all, Resps_all, time_factor_all, attempts_factor_all, hints_factor_all
# *train_data
def train_one_epoch(enhanceModel, optimizer, criterion,
                    batch_size, Topics_all, Resps_all,
                    time_factor_all, attempts_factor_all, hints_factor_all, skill_all, spend_all, mask_all):
    enhanceModel.train()
    all_pred = []
    all_target = []
    n = len(Topics_all) // batch_size
    shuffled_ind = np.arange(len(Topics_all))
    np.random.shuffle(shuffled_ind)
    Topics_all = Topics_all[shuffled_ind]
    Resps_all = Resps_all[shuffled_ind]
    time_factor_all = time_factor_all[shuffled_ind]
    attempts_factor_all = attempts_factor_all[shuffled_ind]
    hints_factor_all = hints_factor_all[shuffled_ind]
    
    skill_all = skill_all[shuffled_ind]
    spend_all = spend_all[shuffled_ind]
    mask_all = mask_all[shuffled_ind]
    # FKT
    y_criterion = nn.BCELoss()
    time_criterion = nn.CrossEntropyLoss()
    
    for idx in tqdm(range(n)):
        optimizer.zero_grad()

        Topics = Topics_all[idx * batch_size: (idx + 1) * batch_size, :]
        Resps = Resps_all[idx * batch_size: (idx + 1) * batch_size, :]
        time_factor = time_factor_all[idx * batch_size:
                                      (idx + 1) * batch_size, :]
        attempts_factor = attempts_factor_all[idx * batch_size:
                                              (idx + 1) * batch_size, :]
        hints_factor = hints_factor_all[idx * batch_size:
                                        (idx + 1) * batch_size, :]
        
        skills = skill_all[idx * batch_size:(idx + 1) * batch_size, :]
        spends = spend_all[idx * batch_size:(idx + 1) * batch_size, :]
        masks = mask_all[idx * batch_size:(idx + 1) * batch_size, :]
        
        input_topics = torch.from_numpy(Topics).long().to(device)
        input_resps = torch.from_numpy(Resps).long().to(device)
        input_time_factor = torch.from_numpy(time_factor).float().to(device)
        input_attempts_factor = torch.from_numpy(attempts_factor).float().to(device)
        input_hints_factor = torch.from_numpy(hints_factor).float().to(device)
        
        input_skill = torch.from_numpy(skills).long().to(device)
        input_spend = torch.from_numpy(spends).long().to(device)
        input_mask = torch.from_numpy(masks).int().to(device)

        y_pred, P, Y, S, out_time, Spend, out2 = enhanceModel(input_topics, input_resps, input_time_factor,
                           input_attempts_factor, input_hints_factor, input_skill, input_spend, input_mask)

        mask = input_topics[:, 1:] > 0
        masked_pred = y_pred[:, 1:][mask]
        masked_truth = input_resps[:, 1:][mask]
        loss_LBKT = criterion(masked_pred, masked_truth.float()).sum()
        # FKT
        index = S == 1
        y_loss = y_criterion(P[index], Y[index].float())
        time_loss = time_criterion(out_time[index], Spend[index])
        loss_FKT = all_loss(bce_loss=y_loss, cross_ent_loss=time_loss)
        
        loss_all = loss_FKT + loss_LBKT
        loss_all.backward()
        print("LBKT_LogisticLoss: %.4f || FKT_LogisticLoss: %.4f || ALL_LogisticLoss: %.4f " % (loss_LBKT, loss_FKT, loss_all))
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        all_pred.append(masked_pred)
        all_target.append(masked_truth)

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    loss = binary_entropy(all_target, all_pred)
    # print("ALL_LogisticLoss: %.6f" % loss)
    auc = compute_auc(all_target, all_pred)
    acc = compute_accuracy(all_target, all_pred)

    return loss, auc, acc


def test_one_epoch(enhanceModel, batch_size, Topics_all, Resps_all,
                   time_factor_all, attempts_factor_all, hints_factor_all,
                   skill_all, spend_all, mask_all):
    enhanceModel.eval()
    # FKT
    y_criterion = nn.BCELoss()
    time_criterion = nn.CrossEntropyLoss()
    y_pred, y_true = [], []
    y_loss = 0.0
    time_loss = 0.0
    threshold = 0.5
    detach = lambda o: o.cpu().detach().numpy().tolist()
    
    all_pred, all_target = [], []
    n = len(Topics_all) // batch_size
    for idx in range(n):
        Topics = Topics_all[idx * batch_size:
                            (idx + 1) * batch_size, :]
        Resps = Resps_all[idx * batch_size:
                          (idx + 1) * batch_size, :]
        time_factor = time_factor_all[idx * batch_size:
                                      (idx + 1) * batch_size, :]
        attempts_factor = attempts_factor_all[idx * batch_size:
                                              (idx + 1) * batch_size, :]
        hints_factor = hints_factor_all[idx * batch_size:
                                        (idx + 1) * batch_size, :]
        
        skills = skill_all[idx * batch_size:(idx + 1) * batch_size, :]
        spends = spend_all[idx * batch_size:(idx + 1) * batch_size, :]
        masks = mask_all[idx * batch_size:(idx + 1) * batch_size, :]
        
        input_topics = torch.from_numpy(Topics).long().to(device)
        input_resps = torch.from_numpy(Resps).long().to(device)
        input_time_factor = torch.from_numpy(time_factor).float().to(device)
        input_attempts_factor = torch.from_numpy(attempts_factor).float().to(device)
        input_hints_factor = torch.from_numpy(hints_factor).float().to(device)
        
        input_skill = torch.from_numpy(skills).long().to(device)
        input_spend = torch.from_numpy(spends).long().to(device)
        input_mask = torch.from_numpy(masks).int().to(device)
        
        with torch.no_grad():
            # y_pred = recurrent(input_topics, input_resps, input_time_factor,
            #                    input_attempts_factor, input_hints_factor)
            yy_pred, P, Y, S, out_time, Spend, out2 = enhanceModel(input_topics, input_resps, input_time_factor,
                                                                  input_attempts_factor, input_hints_factor,
                                                                  input_skill, input_spend, input_mask)
            mask = input_topics[:, 1:] > 0
            masked_pred = yy_pred[:, 1:][mask]
            masked_truth = input_resps[:, 1:][mask]

            masked_pred = masked_pred.detach().cpu().numpy()
            masked_truth = masked_truth.detach().cpu().numpy()

            all_pred.append(masked_pred)
            all_target.append(masked_truth)
            
            # FKT
            index = S == 1
            P, Y = P[index], Y[index].float()
            out_time, Spend = out_time[index], Spend[index]
            y_pred += detach(P)
            y_true += detach(Y)
            y_loss += detach(y_criterion(P, Y) * P.shape[0])
            time_loss += detach(time_criterion(out_time, Spend) * P.shape[0])
            
    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    loss = binary_entropy(all_target, all_pred)
    aucc = compute_auc(all_target, all_pred)
    rmse = mean_squared_error(all_target, all_pred, squared=False)
    acc = compute_accuracy(all_target, all_pred)

    loss_all = all_loss(bce_loss=y_loss, cross_ent_loss=time_loss)
    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    # mse_value = mean_squared_error(y_true, y_pred)
    # mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= threshold else 0 for i in y_pred]
    acc_value = accuracy_score(y_true, bi_y_pred)
    auc_value = auc(fpr, tpr)
    print("loss_fkt: %.6f, acc_fkt: %.6f, auc_fkt: %.6f" % (loss_all, acc_value, auc_value))
    
    return loss, aucc, acc, rmse

# LBKT+FKT
# n_exercises == num_topics
# d_model == dim_tp
def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别

class LKT(KTM):
    def __init__(self, q_num, n_exercises, time_spend, d_model, dim_tp, length, nhead, num_encoder_layers, speed_cate, num_resps, num_units, dropout,
            dim_hidden, memory_size, BATCH_SIZE, q_matrix, seed):
        super(LKT, self).__init__()
        seed_torch(seed)
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        
        # FKT
        # Model_exp(q_num, time_spend,d_model, length, nhead, num_encoder_layers, dropout,speed_cate)
        
        self.enhanceModel = Model_exp(q_num, time_spend, d_model, length, nhead, num_encoder_layers, dropout, speed_cate,
                                      n_exercises, num_resps, num_units, dim_hidden, memory_size, BATCH_SIZE, q_matrix).to(device)
        
        # self.recurrent = Recurrent(n_exercises, dim_tp, num_resps, num_units,
        #                            dropout, dim_hidden, memory_size,
        #                            BATCH_SIZE, q_matrix).to(device)
        
        # print(self.enhanceModel, 'print')
        
        summary(self.enhanceModel)
        self.batch_size = BATCH_SIZE

    def train(self, train_data, test_data, epoch: int,
              lr, lr_decay_step=1, lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.enhanceModel.parameters(), lr=lr,
                                     eps=1e-8, betas=(0.1, 0.999),
                                     weight_decay=1e-6)
        # optimizer = torch.optim.Adam(self.recurrent.parameters(), lr=lr,
        #                              eps=1e-8, betas=(0.1, 0.999),
        #                              weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss(reduction='none')

        best_test_auc = 0
        for idx in range(epoch):
            train_loss, _, _ = train_one_epoch(self.enhanceModel,
                                               optimizer, criterion,
                                               self.batch_size, *train_data)
            # print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
            scheduler.step()
            if test_data is not None:
                _, valid_auc, valid_acc, valid_rmse = self.eval(test_data)
                # print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (
                #     idx, valid_auc, valid_acc, valid_rmse))
                if valid_auc > best_test_auc:
                    best_test_auc = valid_auc
        return best_test_auc

    def eval(self, test_data) -> ...:
        self.enhanceModel.eval()
        return test_one_epoch(self.enhanceModel, self.batch_size, *test_data)

    def save(self, filepath) -> ...:

        torch.save(self.enhanceModel.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.enhanceModel.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

# 模型结构
# ===========================================================================
# Layer (type:depth-idx)                             Param #
# ===========================================================================
# Model_exp                                          --
# ├─EncoderEmbedding: 1-1                            --
# │    └─Embedding: 2-1                              9,088,512
# │    └─Embedding: 2-2                              5,120
# │    └─Embedding: 2-3                              51,200
# │    └─Embedding: 2-4                              1,024
# │    └─Embedding: 2-5                              20,480
# ├─MyTransformerEncoder: 1-2                        --
# │    └─ModuleList: 2-6                             --
# │    │    └─MyTransformerEncoderLayer: 3-1         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-2         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-3         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-4         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-5         1,576,452
# │    │    └─MyTransformerEncoderLayer: 3-6         1,576,452
# │    └─LayerNorm: 2-7                              1,024
# ├─DecoderEmbedding: 1-3                            --
# │    └─Embedding: 2-8                              9,088,512
# │    └─Embedding: 2-9                              88,064
# │    └─Embedding: 2-10                             51,200
# │    └─Embedding: 2-11                             1,024
# │    └─Embedding: 2-12                             20,480
# ├─MyTransformerDecoder: 1-4                        --
# │    └─ModuleList: 2-13                            --
# │    │    └─MyTransformerDecoderLayer: 3-7         2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-8         2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-9         2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-10        2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-11        2,626,581
# │    │    └─MyTransformerDecoderLayer: 3-12        2,626,581
# │    └─LayerNorm: 2-14                             1,024
# ├─LBKTcell: 1-5                                    198,276
# │    └─Layer1: 2-15                                32,896
# │    └─Layer1: 2-16                                32,896
# │    └─Layer1: 2-17                                32,896
# │    └─Linear: 2-18                                52,096
# │    └─Dropout: 2-19                               --
# │    └─Linear: 2-20                                82,048
# │    └─Sigmoid: 2-21                               --
# ├─Linear: 1-6                                      9,106,263
# ├─Sequential: 1-7                                  --
# │    └─Linear: 2-22                                262,400
# │    └─ReLU: 2-23                                  --
# │    └─Linear: 2-24                                10,280
# ├─Sequential: 1-8                                  --
# │    └─Linear: 2-25                                41
# │    └─Sigmoid: 2-26                               --
# ├─Sequential: 1-9                                  --
# │    └─Linear: 2-27                                65,664
# │    └─ReLU: 2-28                                  --
# │    └─Dropout: 2-29                               --
# │    └─Linear: 2-30                                1,290
# ├─Linear: 1-10                                     5,632
# ├─LayerNorm: 1-11                                  1,024
# ├─Sigmoid: 1-12                                    --
# ===========================================================================
# Total params: 53,519,564
# Trainable params: 53,519,564
# Non-trainable params: 0
# ===========================================================================
