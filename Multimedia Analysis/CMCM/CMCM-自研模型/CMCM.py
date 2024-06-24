import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.io
import numpy as np
import time
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn import metrics
import random
import argparse

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) #(7680, 8, 9, 9)
        n_head = attn.shape[1] #8
        batch_size = q.shape[0] # 7680
        len_q = q.shape[-2] #9
        len_k = k.shape[-2] #9
        mask= mask.unsqueeze(1) # (128,1,1, 66)
        mask= mask.expand(-1, n_head, -1, -1) # (7680, 8, 1, 9)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.  #(7680, 1, 9)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, input_q, input_k, input_v, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            input_q, input_k, input_v, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Self_Transformer(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, win_len, d_word_vec, d_model, timelen, n_layers=1, n_head=8, d_k=64, d_v=64,
             d_inner=2048, dropout=0.1,  scale_emb=False):

        super().__init__()
        self.position_enc = PositionalEncoding(d_word_vec, n_position = win_len + 2)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.win_len = win_len
        self.cls_fea = nn.Parameter(torch.ones(d_model), requires_grad=True) #(1184)
        self.cls_pos = nn.Parameter(torch.ones((timelen,d_model)), requires_grad=True) #(60,1184)
    def forward(self, src_seq, src_mask, return_attns=False):

        #print(src_seq.shape) #(128, 66, 1184)
        #print(src_mask.shape) #(128, 66)
        batch_size = src_seq.shape[0]
        string_gpu = "cuda:" + str(gpu_num)
        device = torch.device(string_gpu if torch.cuda.is_available() else "cpu")
        win_fea = True
        win_mask = True
        for i in range(src_seq.shape[0]):
            for j in range(src_seq.shape[1]-self.win_len+1):
                win_seq = src_seq[i, j:j+self.win_len, :] #(7, 1184)
                cls_pos = self.cls_pos[j, :].unsqueeze(0) #(1, 1184)
                cls_fea = self.cls_fea.unsqueeze(0) #(1, 1184)
                win_seq = torch.cat((cls_fea, cls_pos, win_seq), 0) #(9, 1184)

                mask = src_mask[i, j:j+self.win_len] #(7)
                mask = torch.cat((torch.ones(1).to(device), torch.ones(1).to(device), mask), 0)  # (9)

                if i==0 and j==0:
                    win_fea = win_seq.unsqueeze(0)
                    win_mask = mask.unsqueeze(0)
                else:
                    win_fea = torch.cat((win_fea, win_seq.unsqueeze(0)), 0)
                    win_mask = torch.cat((win_mask, mask.unsqueeze(0)), 0)
        enc_slf_attn_list = []
        # print("1", win_fea.shape)  #(7680, 9, 1184)
        # print("1", win_mask.shape)  # (7680, 9)
        # -- Forward
        enc_output = win_fea
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, enc_output, enc_output, slf_attn_mask=win_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        #print(enc_output.shape) #(7680, 9, 1184)
        enc_output = enc_output[:, 0 , :]  #(7680, 1184) 每个序列只取第一个CLS TOKEN
        enc_output = enc_output.reshape(batch_size, -1, enc_output.shape[-1]) #(128, 60, 1184)


        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Cross_Transformer(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, timelen, win_len, dim_q, dim_k, dim,  n_layers=1, n_head=8, d_k=64, d_v=64,
             d_inner=2048, dropout=0.1,  scale_emb=False):

        super().__init__()

        self.fc1 = nn.Linear(dim_q, dim, bias=False)
        self.fc2 = nn.Linear(dim_k, dim, bias=False)

        self.position_enc = PositionalEncoding(dim, n_position = win_len + 2)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.scale_emb = scale_emb
        self.win_len = win_len
        self.timelen = timelen
        self.dim = dim


        self.fc = nn.Linear(timelen * dim, 1, bias=False)

    def forward(self, q_seq, k_seq, k_mask, return_attns=False):

        #print(src_seq.shape) #(128, 66, 1184)
        #print(src_mask.shape) #(128, 66)
        q_seq = self.fc1(q_seq)  # (128, 60, 512)
        k_seq = self.fc2(k_seq) # (128, 60, 512)
        win_q = q_seq.reshape(-1, 1, q_seq.shape[-1])  #(7680, 1, 512)
        batch_size = q_seq.shape[0]
        string_gpu = "cuda:" + str(gpu_num)
        device = torch.device(string_gpu if torch.cuda.is_available() else "cpu")

        # 给Key数据前后加padding
        k_padding = torch.from_numpy(np.zeros((batch_size, int(self.win_len / 2), k_seq.shape[-1]))).float().to(device)  # (128, 3, 1024/160)
        k_seq = torch.cat((k_padding, k_seq, k_padding), 1)  # (128, 66, 512)

        for i in range(batch_size):
            for j in range(self.timelen):
                win_k_seq = k_seq[i, j:j+self.win_len, :] #(7, 512)
                win_k_mask = k_mask[i, j:j+self.win_len] #(7)

                if i==0 and j==0:
                    win_k = win_k_seq.unsqueeze(0)
                    win_mask = win_k_mask.unsqueeze(0)
                else:
                    win_k = torch.cat((win_k, win_k_seq.unsqueeze(0)), 0)
                    win_mask = torch.cat((win_mask, win_k_mask.unsqueeze(0)), 0)
        enc_slf_attn_list = []
        #print("1", win_k.shape)  #(7680, 7, 512)
        #print("1", win_mask.shape)  # (7680, 7)
        # -- Forward
        enc_output = win_k
        if self.scale_emb:
            win_q *= self.dim ** 0.5
            win_k *= self.dim ** 0.5
        win_q = self.dropout(self.position_enc(win_q))
        win_q = self.layer_norm(win_q)
        win_k = self.dropout(self.position_enc(win_k))
        win_k = self.layer_norm(win_k)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(win_q, win_k, win_k, slf_attn_mask=win_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        #print("111", enc_output.shape) #(7680, 1, 512)

        enc_output = enc_output.reshape(-1, self.timelen, enc_output.shape[-1])  # (128, 60, 512)
        #enc_output = enc_output.reshape(batch_size, -1) #(128, 60*512)
        #enc_output = self.fc(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,



class Dist_metric(nn.Module):
    def __init__(self):
        super(Dist_metric, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(1184, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x1, x2):
        temp = torch.cat((x1, x2), 1)
        weight = self.fc1(temp)
        return weight


class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2)
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return out


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1184, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.fc_a = nn.Sequential(
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )
        self.fc_b = nn.Sequential(
            nn.Linear(160, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid()
        )
        self.modal_weight = nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, x1, x2):
        weight = torch.empty(x1.size(0), x1.size(1))
        for i in range(x1.size(1)):
            temp = torch.cat((torch.squeeze(x1[:, i, :]), torch.squeeze(x2[:, i, :])), 1)
            weight0 = torch.squeeze(self.fc1(temp))
            weight0 = F.softmax(weight0)
            weight[:, i] = weight0[:, 1]
        weight = torch.unsqueeze(F.softmax(weight, dim=1), 2).to(device)
        x1 = torch.sum(x1 * weight, dim=1)
        x2 = torch.sum(x2 * weight, dim=1)
        output1 = self.fc_a(x1)
        output2 = self.fc_b(x2)
        self.modal_weight.data = F.softmax(self.modal_weight, dim=0)
        output = output1 * self.modal_weight[0] + output2 * self.modal_weight[1]
        return output


def cross_entropy_loss(output, target):
    return -torch.sum(output.log() * target) / output.shape[0]


def DTWDistance(s1, s2):
    DTW = {}

    for i in range(s1.shape[0]):
        DTW[(i, -1)] = float('inf')
    for i in range(s2.shape[0]):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(s1.shape[0]):
        for j in range(s2.shape[0]):
            dist = torch.norm(s1[i].sub(s2[j]))
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return torch.sqrt(DTW[len(s1) - 1, len(s2) - 1])

def data_normal_2d(orign_data,dim="col"):
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    return norm_data

def subset(alist, idxs):
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])
    return sub_list

def split_list(alist, group_num=10, shuffle=True, retain_left=False):
    index = list(range(len(alist)))

    if shuffle:
        random.shuffle(index)

    elem_num = len(alist) // group_num
    sub_lists = []

    for idx in range(group_num):
        start, end = idx * elem_num, (idx + 1) * elem_num
        sub_lists.append(subset(alist, index[start:end]))

    if retain_left and group_num * elem_num != len(index):
        sub_lists.append(subset(alist, index[end:]))

    return sub_lists

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=0, help='Get the GPU_NUM')
parser.add_argument('--win_len', type=int, default=7, help='Get the maxleft')
parser.add_argument('--outputdim', type=int, default=1024, help='Get the maxright')
args = parser.parse_args()
gpu_num = args.gpu_num
win_len = args.win_len
outputdim = args.outputdim
print("gpu_num: ", gpu_num)
print("win_len: ", win_len)
print("outputdim: ", outputdim)


batch_size = 360
num_epoch1 = 200
num_epoch2 = 1
num_fold = 10
divide_num = 6

string_gpu = "cuda:" + str(gpu_num)
device = torch.device(string_gpu if torch.cuda.is_available() else "cpu")

# 读取外周生理信号数据
datapath1 = '/data1/eegdata/Deap/处理后数据/deapperipheral_divide_by_5.mat'
X1 = scipy.io.loadmat(datapath1)['X']
Y = scipy.io.loadmat(datapath1)['Y']

a = torch.Tensor(X1)
a = a.contiguous().view(-1, a.size(-2), a.size(-1))
X1 = a.numpy()  # (1280, 8, 7680)
peridata = X1  # ALL
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        temp = X1[i, j, :].reshape(-1, 1)
        temp = preprocessing.MaxAbsScaler().fit_transform(temp)
        peridata[i, j, :] = np.squeeze(temp)

# for i in range(8):
# 	print(numpy.max(peridata[:,i,:]))
# 	print(numpy.min(peridata[:, i, :]))
# 	print(numpy.mean(peridata[:, i, :]))

b = torch.Tensor(Y)
b = b.contiguous().view(-1, b.size(-1))
b = b.repeat_interleave(divide_num, dim=0)
Y = b.numpy()  # (12800, 4)

a = torch.Tensor(peridata)  # (1280, 8, 7680)
a = a.permute(0, 2, 1)
a = a.contiguous().view(a.size(0), 60, -1, a.size(-1))
a = a.contiguous().view(a.size(0), 60, -1)
a = a.contiguous().view(a.size(0), divide_num, -1, a.size(-1))
a = a.contiguous().view(-1, a.size(-2), a.size(-1))
# a = torch.unsqueeze(a, dim=1)
peridata = a.numpy()
#print(peridata.shape) #(12800, 6, 1024)


# 读取脑电信号
datapath2 = '/data1/eegdata/Deap/处理后数据/deapExtractedFeatures_divide_by_5.mat'
X2 = scipy.io.loadmat(datapath2)['X']

a = torch.Tensor(X2)
a = a.permute(0, 1, 3, 2, 4)
a = a.contiguous().view(a.size(0), a.size(1), a.size(2), -1)  # (32,40,60,160)
a = a.contiguous().view(-1, a.size(-2), a.size(-1))
a = a.contiguous().view(a.size(0), divide_num, -1, a.size(-1))
a = a.contiguous().view(-1, a.size(-2), a.size(-1)) # (12800, 6, 160)
# a = torch.unsqueeze(a, dim=1)
eegdata = a.numpy()
#print(eegdata.shape)  # (12800, 6, 160)
# for i in range(1280):
# 	print(numpy.max(eegdata[i, :, :]))
# 	print(numpy.min(eegdata[i, :, :]))
# 	print(numpy.mean(eegdata[i, :, :]))

X_all = np.concatenate((peridata[:, :, :], eegdata[:, :, :]), 2)  # (12800, 6, 1184)
Y_all = np.squeeze(Y[:, 0])  # (1280,) #DEAP O:vlaence 1:arousal

#给数据前后加padding
X_len = X_all.shape[1] # 6
X_num = X_all.shape[0] # 12800
X_padding = np.zeros((X_num, int(win_len/2), X_all.shape[-1])) #(12800, 1, 1184)
X_all = np.concatenate((X_padding, X_all, X_padding), 1) #(12800, 8, 1184)

#mask矩阵生成
Mask_all = np.concatenate((np.ones((X_num, int(win_len/2))), np.zeros((X_num, X_len)), np.ones((X_num, int(win_len/2)))), 1) #(12800, 66)

subject_test_acc_list = []
subject_test_f1_list = []
subject_test_acc = 0
subject_test_f1 = 0
subject_printscore_list = []
#
for Subject_num in list([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]):
#for Subject_num in list([1,2]):
    print("Subject "+ str(Subject_num) + "-------------------------")
    fold_test_acc_list = []
    fold_printscore_list = []
    fold_test_acc = 0
    all = [i for i in range(40 * divide_num * (Subject_num - 1), 40* divide_num * Subject_num)]
    splitfoldlist = split_list(all, group_num = num_fold)
    prelist_sub = []
    truelist_sub = []
    printscore_sub = 0
    for fold in range(num_fold):
        print("Fold " + str(fold + 1) + "----------------------------")
        print(time.asctime(time.localtime(time.time())))
        prelist_fold = []
        truelist_fold = []
        printscore_fold = []
        list2 = splitfoldlist[fold]
        list1 = [i for i in all if i not in list2]

        Xtrain = X_all[list1, :, :]# (360, 8, 1184)
        Ytrain = Y_all[list1]  # (360,) #DEAP O:vlaence 1:arousal
        Masktrain = Mask_all[list1, :]
        Xtest = X_all[list2, :, :]  # (40, 8, 1184)
        Ytest = Y_all[list2]  # (40,)  #DEAP O:vlaence 1:arousal
        Masktest = Mask_all[list2, :]

        train_data = TensorDataset(torch.tensor(Xtrain, dtype=torch.float32),
                                torch.tensor(Ytrain, dtype=torch.float32),
                                torch.tensor(Masktrain, dtype=torch.float32))   # 训练的数据集
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
        test_data = TensorDataset(torch.tensor(Xtest, dtype=torch.float32),
                                torch.tensor(Ytest, dtype=torch.float32),
                                torch.tensor(Masktest, dtype=torch.float32))
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        len_train_data = len(train_data)
        len_test_data = len(test_data)

        # ------------------------------------------------------------------------------------------------------#
        #print('Transformer Test...')
        dim_per = peridata.shape[-1] #1024
        dim_eeg = eegdata.shape[-1] #160
        transformer_per = Self_Transformer(win_len, dim_per, dim_per, X_len).to(device)
        transformer_eeg = Self_Transformer(win_len, dim_eeg, dim_eeg, X_len).to(device)
        transformer_per_eeg = Cross_Transformer(X_len, win_len, dim_per, dim_eeg, outputdim).to(device)
        transformer_eeg_per = Cross_Transformer(X_len, win_len, dim_eeg, dim_per, outputdim).to(device)
        fc_output = nn.Linear(outputdim*2, 1, bias=False).to(device)

        optimizer = torch.optim.Adam([{'params': transformer_per.parameters()},{'params': transformer_eeg.parameters()},{'params': fc_output.parameters()}],lr=1e-3)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss_func = nn.CrossEntropyLoss()

        maxtestacc = -1
        for epoch in range(num_epoch1):
            print(f'Epoch {epoch + 1}/{num_epoch1}:')
            #print(time.asctime(time.localtime(time.time())))
            train_loss = .0
            train_acc = .0
            trainstepsum = 0
            for trainstep, (batch_x, batch_y, batch_mask) in enumerate(train_loader):
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)

                out_per1 = transformer_per(batch_x[:, :, :dim_per], batch_mask)[0] # (128, 60, 1024)
                out_eeg1 = transformer_eeg(batch_x[:, :, dim_per:], batch_mask)[0] # (128, 60, 160)
                #print("Self-transformer Finished!")
                #print(time.asctime(time.localtime(time.time())))
                out_per2 = transformer_per_eeg(out_per1, out_eeg1, batch_mask)
                out_eeg2 = transformer_eeg_per(out_eeg1, out_per1, batch_mask)
                #print("Cross-transformer Finished!")
                #print(time.asctime(time.localtime(time.time())))
                # #求一致性分数(闵可夫斯基距离)
                out_perdata  = out_per2[0]  #(128, 60, 512)
                out_eegdata = out_eeg2[0]   #(128, 60, 512)
                temp = out_perdata.sub(out_eegdata) #(128, 60, 512)
                #temp_padding = torch.from_numpy(np.zeros((temp.size(0), int(win_len / 2), temp.shape[-1]))).float().to(device)  # (128, 3, 512)
                temp_padding_left = temp[:, :int(win_len / 2), :]
                temp_padding_left = torch.mean(temp_padding_left, dim=1, keepdim=True)
                temp_padding_left= temp_padding_left.expand([temp_padding_left.size(0), int(win_len / 2), temp_padding_left.size(-1)])
                temp_padding_right = temp[:, -1 * int(win_len / 2):, :]
                temp_padding_right = torch.mean(temp_padding_right, dim=1, keepdim=True)
                temp_padding_right = temp_padding_right.expand([temp_padding_right.size(0), int(win_len / 2), temp_padding_right.size(-1)])
                temp = torch.cat((temp_padding_left, temp, temp_padding_right), 1)  # (128, 66, 512)

                for j in range(X_len):
                    win_temp = temp[:, j : j + win_len, :] #(128, 7, 512)

                    if j==0:
                        score = win_temp.unsqueeze(1) #(128, 1, 7, 512)
                    else:
                        score = torch.cat((score, win_temp.unsqueeze(1)), 1)
                #print(score.shape)  #(128, 60, 7, 512)
                score = score.reshape(score.shape[0], score.shape[1], -1) #(128, 60, 3584)
                score = torch.norm(score, dim=2) #(128, 60)
                #score1 = F.softmax(score, dim=-1)  # (128, 60)
                score= data_normal_2d(score)
                score = score * -1
                score = score.add(1)
                scoresum = (torch.sum(score, 1)).unsqueeze(1)
                score = torch.div(score, scoresum)
                score = score.unsqueeze(-1)  #(128, 60, 1)
                score = score.expand(-1, -1, outputdim) #(128, 60, 512)

                # 求一致性分数(DWT)
                # out_perdata = out_per2[0]  # (128, 60, 512)
                # out_eegdata = out_eeg2[0]  # (128, 60, 512)
                #
                #
                #
                # temp_padding = torch.from_numpy(np.zeros((batch_size, int(win_len / 2), outputdim))).float().to(device)  # (128, 3, 512)
                # out_perdata_padding = torch.cat((temp_padding, out_perdata, temp_padding), 1)  # (128, 66, 512)
                # out_eegdata_padding = torch.cat((temp_padding, out_eegdata, temp_padding), 1)  # (128, 66, 512)
                #
                # score = torch.FloatTensor(batch_size, X_len).to(device)
                # for i in range(batch_size):
                #     for j in range(X_len):
                #         win_perdata = out_perdata_padding[i, j: j + win_len, :]  # (7, 512)
                #         win_eegdata = out_eegdata_padding[i, j: j + win_len, :]  # (7, 512)
                #         score[i, j] = DTWDistance(win_perdata, win_eegdata)  #(128, 60)
                #
                # score = F.softmax(score, dim=-1)  # (128, 60)
                # score = score.unsqueeze(-1)  # (128, 60, 1)
                # score = score.expand(-1, -1, outputdim)  # (128, 60, 512)

                #print("Credit Score Finished!")
                #print(time.asctime(time.localtime(time.time())))
                #加权求和
                out_perdata = torch.mul(out_perdata, score)  #(128, 60, 512)
                out_eegdata = torch.mul(out_eegdata, score)  #(128, 60, 512)
                out_perdata = torch.sum(out_perdata, dim=1)  #(128, 512)
                out_eegdata = torch.sum(out_eegdata, dim=1)  #(128, 512)
                out = fc_output(torch.cat((out_perdata, out_eegdata), 1))  #(128, 1)
                loss = criterion(out.squeeze(), batch_y)
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.data.item()
                train_acc += ((torch.sigmoid(out.squeeze()) >= 0.5) == batch_y).sum().data.item()
                trainstepsum = trainstepsum + 1

            print('Train loss: {:.6f}, Train acc: {:.6f}'.format(train_loss / trainstepsum, train_acc / len_train_data))

            test_loss = .0
            test_acc = .0
            teststepsum = 0
            with torch.no_grad():
                prelist_temp = []
                turelist_temp = []
                printscore_temp = 0
                for teststep, (batch_x, batch_y, batch_mask) in enumerate(test_loader):
                    batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)

                    out_per1 = transformer_per(batch_x[:, :, :dim_per], batch_mask)[0]  # (128, 60, 1024)
                    out_eeg1 = transformer_eeg(batch_x[:, :, dim_per:], batch_mask)[0]  # (128, 60, 160)
                    # print("Self-transformer Finished!")
                    # print(time.asctime(time.localtime(time.time())))
                    out_per2 = transformer_per_eeg(out_per1, out_eeg1, batch_mask)
                    out_eeg2 = transformer_eeg_per(out_eeg1, out_per1, batch_mask)
                    # print("Cross-transformer Finished!")
                    # print(time.asctime(time.localtime(time.time())))
                    # #求一致性分数(闵可夫斯基距离)
                    out_perdata = out_per2[0]  # (128, 60, 512)
                    out_eegdata = out_eeg2[0]  # (128, 60, 512)
                    temp = out_perdata.sub(out_eegdata)  # (128, 60, 512)
                    temp_padding = torch.from_numpy(np.zeros((temp.size(0), int(win_len / 2), temp.shape[-1]))).float().to(
                        device)  # (128, 3, 512)
                    temp = torch.cat((temp_padding, temp, temp_padding), 1)  # (128, 66, 512)

                    for j in range(X_len):
                        win_temp = temp[:, j: j + win_len, :]  # (128, 7, 512)

                        if j == 0:
                            score = win_temp.unsqueeze(1)  # (128, 1, 7, 512)
                        else:
                            score = torch.cat((score, win_temp.unsqueeze(1)), 1)
                    # print(score.shape)  #(128, 60, 7, 512)
                    score = score.reshape(score.shape[0], score.shape[1], -1)  # (128, 60, 3584)
                    score = torch.norm(score, dim=2)  # (128, 60)
                    printscore = torch.sum(score, dim=1)
                    printscore = torch.div(printscore, score.shape[1])
                    printscore_temp = printscore
                    score = F.softmax(score, dim=-1)  # (128, 60)
                    score = score.unsqueeze(-1)  # (128, 60, 1)
                    score = score.expand(-1, -1, outputdim)  # (128, 60, 512)

                    # 求一致性分数(DWT)
                    # out_perdata = out_per2[0]  # (128, 60, 512)
                    # out_eegdata = out_eeg2[0]  # (128, 60, 512)
                    #
                    #
                    #
                    # temp_padding = torch.from_numpy(np.zeros((batch_size, int(win_len / 2), outputdim))).float().to(device)  # (128, 3, 512)
                    # out_perdata_padding = torch.cat((temp_padding, out_perdata, temp_padding), 1)  # (128, 66, 512)
                    # out_eegdata_padding = torch.cat((temp_padding, out_eegdata, temp_padding), 1)  # (128, 66, 512)
                    #
                    # score = torch.FloatTensor(batch_size, Xtrain_len).to(device)
                    # for i in range(batch_size):
                    #     for j in range(Xtrain_len):
                    #         win_perdata = out_perdata_padding[i, j: j + win_len, :]  # (7, 512)
                    #         win_eegdata = out_eegdata_padding[i, j: j + win_len, :]  # (7, 512)
                    #         score[i, j] = DTWDistance(win_perdata, win_eegdata)  #(128, 60)
                    #
                    # score = F.softmax(score, dim=-1)  # (128, 60)
                    # score = score.unsqueeze(-1)  # (128, 60, 1)
                    # score = score.expand(-1, -1, outputdim)  # (128, 60, 512)

                    # print("Credit Score Finished!")
                    # print(time.asctime(time.localtime(time.time())))
                    # 加权求和
                    out_perdata = torch.mul(out_perdata, score)  # (128, 60, 512)
                    out_eegdata = torch.mul(out_eegdata, score)  # (128, 60, 512)
                    out_perdata = torch.sum(out_perdata, dim=1)  # (128, 512)
                    out_eegdata = torch.sum(out_eegdata, dim=1)  # (128, 512)
                    out = fc_output(torch.cat((out_perdata, out_eegdata), 1))  # (128, 1)
                    loss = criterion(out.squeeze(), batch_y)
                    # print(loss)
                    test_loss += loss.data.item()
                    test_acc += ((torch.sigmoid(out.squeeze()) >= 0.5) == batch_y).sum().data.item()
                    teststepsum = teststepsum + 1
                    testpred_y = list(((torch.sigmoid(out.squeeze()) >= 0.5) + 0).cpu().numpy())
                    prelist_temp.extend(testpred_y)
                    turelist_temp.extend(list(batch_y.cpu().numpy()))
                print('Test loss: {:.6f}, Test acc: {:.6f}'.format(test_loss / teststepsum, test_acc / len_test_data))
                if test_acc / len_test_data > maxtestacc:
                    maxtestacc = test_acc / len_test_data
                    prelist_fold = prelist_temp
                    printscore_fold = printscore_temp
        print("Subject " + str(Subject_num) + " Fold " + str(fold + 1) + " maxtest:  ", maxtestacc)
        fold_test_acc_list.append(maxtestacc)
        fold_printscore_list.append(printscore_fold)
        fold_test_acc += maxtestacc
        truelist_fold = [int(dd) for dd in turelist_temp]
        truelist_sub.extend(truelist_fold)
        prelist_sub.extend(prelist_fold)
        printscore_sub = 0
        for kk in range(len(fold_printscore_list)):
            if kk == 0:
                printscore_sub = fold_printscore_list[-1]
            else:
                printscore_sub = torch.cat([printscore_sub, fold_printscore_list[-(kk + 1)]], dim=0)
        printscore_sub = printscore_sub.cpu().numpy()
    print("Subject " + str(Subject_num) + " results-------------------------------------\n\n")
    print("Average score:", printscore_sub.sum() / len(printscore_sub))
    print("Truelist:")
    print(truelist_sub)
    print("Prelist:")
    print(prelist_sub)
    accacc = metrics.accuracy_score(truelist_sub, prelist_sub)
    print('acc: ' + str(accacc))
    f1 = metrics.f1_score(truelist_sub, prelist_sub, average='macro')
    print("Subject " + str(Subject_num) + " fold_test_acc_list: ", fold_test_acc_list)
    print("Subject " + str(Subject_num) + " final average acc:  ", fold_test_acc / num_fold)
    print("Subject " + str(Subject_num) + " final average f1:  ", f1)
    subject_test_acc_list.append(fold_test_acc / num_fold)
    subject_test_f1_list.append(f1)
    print("Now subject_test_acc_list", subject_test_acc_list)
    print("Now subject_test_f1_list", subject_test_f1_list)
    print("-------------------------------------\n\n")
    subject_test_acc += fold_test_acc / num_fold
    subject_test_f1 += f1
    subject_printscore_list.append(printscore_sub)
print("Final subject_test_acc_list: ", subject_test_acc_list)
print("Final subject_test_f1_list", subject_test_f1_list)
print("Final average subject_test_acc:  ", subject_test_acc / len(subject_test_acc_list))
print("Final average subject_test_f1:  ", subject_test_f1 / len(subject_test_f1_list))
savepath = 'deap_valence_score.mat'
scipy.io.savemat(savepath, {'S': subject_printscore_list})
# # ------------------------------------------------------------------------------------------------------#
# print('\n\nPretrain weight...')
# dist_m = Dist_metric().to(device)
# optimizer = torch.optim.Adam(dist_m.parameters())
# criterion = nn.BCEWithLogitsLoss(reduction='mean')
# loss_func = nn.CrossEntropyLoss()
# maxacc = 0
# for epoch in range(num_epoch1):
#     print(f'Epoch {epoch + 1}/{num_epoch1}:')
#     train_loss = .0
#     train_acc = .0
#     trainstepsum = 0
#
#     for trainstep, (batch_x, batch_y) in enumerate(train_loader):
#         x1 = batch_x[:, :, 0:1024]
#         x2 = batch_x[:, :, 1024:]
#         x0 = torch.cat((x1[:, 30:, :], x1[:, 0:30, :]), 1)
#         x0 = x0.contiguous().view(-1, x0.size(-1))
#         x1 = x1.contiguous().view(-1, x1.size(-1))
#         x2 = x2.contiguous().view(-1, x2.size(-1))
#         batch_x1 = torch.cat((x0, x1), 0)  # [15360, 1024]
#         batch_x2 = torch.cat((x2, x2), 0)  # [15360, 160]
#         y0 = torch.zeros(x0.size(0), dtype=torch.long)
#         y1 = torch.ones(x1.size(0), dtype=torch.long)
#         batch_y = torch.cat((y0, y1), 0)
#         batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
#         out = torch.squeeze(dist_m(batch_x1, batch_x2))
#         # print(out.shape)
#         # print(batch_y.shape)
#         loss = loss_func(out, batch_y)
#         # print(loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         trainpred_y = (torch.max(out, 1)[1]).cpu().numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
#         accuracy0 = float((trainpred_y == batch_y.cpu().numpy()).astype(int).sum()) / float(batch_y.size(0))
#         train_acc = train_acc + accuracy0
#         trainstepsum = trainstepsum + 1
#     train_acc = train_acc / trainstepsum
#     print('| train loss: %.4f' % loss.data.cpu().numpy(), '| train accuracy: %.2f' % train_acc)
#     if round(train_acc, 2) > maxacc:
#         string_pkl = "./pretrain/epoch" + str(epoch + 1) + ".pkl"
#         torch.save(dist_m.state_dict(), string_pkl)
#         maxacc = round(train_acc, 2)
# print(time.asctime(time.localtime(time.time())))
#
# all = [i for i in range(a.size(0))]
# student_average_acc = 0
# student_acc_list = []
# teacher_average_acc = 0
# teacher_acc_list = []
# student_with_teacher_average_acc = 0
# student_with_teacher_acc_list = []
# for fold in range(num_fold):
#     print("\n\n\n\nThis is Fold!!!!!!!!!!------------------------------------", str(fold + 1))
#     list1 = []
#     list2 = []
#     for i in range(32):
#         for j in range(4):
#             list2.append(40 * i + 4 * fold + j)
#     list1 = [i for i in all if i not in list2]
#
#     Xtrain = np.concatenate((peridata[list1, :, :], eegdata[list1, :, :]), 2)  # (1152, 60, 1184)
#     Ytrain = np.squeeze(Y[list1, 0])  # (1152,) #DEAP O:vlaence 1:arousal
#     Xtest = np.concatenate((peridata[list2, :, :], eegdata[list2, :, :]), 2)  # (128, 60, 1184)
#     Ytest = np.squeeze(Y[list2, 0])  # (128,)  #DEAP O:vlaence 1:arousal
#
#     train_data = TensorDataset(torch.tensor(Xtrain, dtype=torch.float32),
#                                torch.tensor(Ytrain, dtype=torch.float32))  # 训练的数据集
#     train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=False)
#     test_data = TensorDataset(torch.tensor(Xtest, dtype=torch.float32), torch.tensor(Ytest, dtype=torch.float32))
#     test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False)
#     len_train_data = len(train_data)
#     len_test_data = len(test_data)
#
#     # ------------------------------------------------------------------------------------------------------#
#
#     print('\n\nTraining teacher...')
#     teacher = Teacher().to(device)
#     teacher_dict = teacher.state_dict()
#     pre_dict = {k: v for k, v in dist_m.state_dict().items() if k in teacher_dict}
#     teacher_dict.update(pre_dict)
#     teacher.load_state_dict(teacher_dict)
#     optimizer = torch.optim.Adam(teacher.parameters())
#     criterion = nn.BCEWithLogitsLoss()
#     Best2 = 0
#     for epoch in range(num_epoch2):
#         print(f'Epoch {epoch + 1}/{num_epoch2}:')
#
#         train_loss = .0
#         train_acc = .0
#         teacher.train()
#         trainstepsum = 0
#         for trainstep, (batch_x, batch_y) in enumerate(train_loader):
#             batch_x1 = batch_x[:, :, 0:1024]
#             batch_x2 = batch_x[:, :, 1024:]
#             batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
#
#             out = torch.squeeze(teacher(batch_x1, batch_x2))
#             loss = criterion(out, batch_y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.data.item()
#             train_acc += ((torch.sigmoid(out) >= 0.5) == batch_y).sum().data.item()
#             trainstepsum = trainstepsum + 1
#         print('Train loss: {:.6f}, Train acc: {:.6f}'.format(train_loss / trainstepsum, train_acc / len_train_data))
#
#         eval_loss = .0
#         eval_acc = .0
#         teacher.eval()
#         with torch.no_grad():
#             teststepsum = 0
#             for trainstep, (batch_x, batch_y) in enumerate(test_loader):
#                 batch_x1 = batch_x[:, :, 0:1024]
#                 batch_x2 = batch_x[:, :, 1024:]
#                 batch_x1, batch_x2, batch_y = batch_x1.to(device), batch_x2.to(device), batch_y.to(device)
#
#                 out = torch.squeeze(teacher(batch_x1, batch_x2))
#                 loss = criterion(out, batch_y)
#
#                 eval_loss += loss.data.item()
#                 eval_acc += ((torch.sigmoid(out) >= 0.5) == batch_y).sum().data.item()
#                 teststepsum = teststepsum + 1
#             print('Eval loss: {:.6f}, Eval acc: {:.6f}'.format(eval_loss / teststepsum, eval_acc / len_test_data))
#             if (eval_acc / len_test_data) > Best2:
#                 Best2 = eval_acc / len_test_data
#     print("Fold:", str(fold + 1), "teacher Eval Best Acc:", Best2)
#     teacher_average_acc = teacher_average_acc + Best2
#     teacher_acc_list.append(Best2)
#
# # ------------------------------------------------------------------------------------------------------#
# teacher_average_acc = teacher_average_acc / num_fold
# # student_average_acc = student_average_acc / num_fold
# # student_with_teacher_average_acc = student_with_teacher_average_acc / num_fold
# print("teacher_acc_list:", teacher_acc_list)
# # print("student_acc_list:", student_acc_list)
# # print("student_with_teacher_acc_list:", student_with_teacher_acc_list)
# print("---------------------------------------------------------------------")
# print("teacher_average_acc:", teacher_average_acc)
# # print("student_average_acc:", student_average_acc)
# # print("student_with_teacher_average_acc:", student_with_teacher_average_acc)
# print(time.asctime(time.localtime(time.time())))
