import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats

from .global_configs import *
import math

class CME(nn.Module):
    def __init__(self):
        super(CME, self).__init__()
        # 索引序列用下面两行先转化为向量序列
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.layernorm1 = nn.LayerNorm(TEXT_DIM)
        self.layernorm2 = nn.LayerNorm(TEXT_DIM)
        self.hv = SelfAttention(TEXT_DIM)
        self.ha = SelfAttention(TEXT_DIM)
        self.aa = SelfAttention(TEXT_DIM)
        self.vv = SelfAttention(TEXT_DIM)

        self.ffn1 = PositionWiseFeedForward(TEXT_DIM)
        self.ffn2 = PositionWiseFeedForward(TEXT_DIM)
        self.ffn3 = PositionWiseFeedForward(TEXT_DIM)
        self.ffn4 = PositionWiseFeedForward(TEXT_DIM)
        self.gate1 = GatedMultimodalLayer(TEXT_DIM, TEXT_DIM)
        
    
    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        visual_ = self.visual_embedding(visual_ids)
        acoustic_ = self.acoustic_embedding(acoustic_ids)

        visual_ = self.hv(text_embedding, visual_) + visual_
        visual_ = self.ffn1(self.vv(visual_, visual_))+ visual_

        acoustic_ = self.ha(text_embedding, acoustic_) + acoustic_
        acoustic_ = self.ffn2(self.aa(acoustic_, acoustic_)) + acoustic_

        shift = self.gate1(visual_, acoustic_)
        shift = shift + self.ffn4(shift)

        text_embedding = text_embedding + self.ffn3(text_embedding)
        embedding_shift = shift + text_embedding

        return embedding_shift, visual_, acoustic_

    
class PositionWiseFeedForward(nn.Module):

    """
    w2(relu(w1(layer_norm(x))+b1))+b2
    """

    def __init__(self, TEXT_DIM, dropout=0.5):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.w_2 = nn.Linear(TEXT_DIM, TEXT_DIM)
        self.layer_norm = nn.LayerNorm(TEXT_DIM, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=16):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        K = self.Wk(embedding)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        context_layer = torch.matmul(weight_prob, V)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    
class GatedMultimodalLayer(nn.Module):
    
    def __init__(self, size_in1, size_in2, size_out=16):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out*2, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        x = torch.cat((h1, h2), dim=2)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        outs = []
        for i in range(z.size()[0]):
            zi = z[i,:,:]
            outi = zi*x1[i,:,:] + (1-zi)*x2[i,:,:]
            outs.append(outi)
        result = torch.stack(outs)
        return result
