
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import copy

class EncoderEmbedding(nn.Module):
    def __init__(self, q_num, time_spend,  length,d_model):
        super(EncoderEmbedding, self).__init__()
        self.seq_len = length
        self.exercise_embed = nn.Embedding(q_num, d_model)
        self.category_embed = nn.Embedding(time_spend, d_model)
        self.position_embed = nn.Embedding(length, d_model)
        self.response_embed = nn.Embedding(2, d_model)
        self.skill_embed = nn.Embedding(40, d_model)

    def forward(self, exercises, categories,response,skill):
        e = self.exercise_embed(exercises)
        c = self.category_embed(categories)
        r=  self.response_embed(response)
        seq = torch.arange(self.seq_len).cuda().unsqueeze(0)
        p = self.position_embed(seq)
        sk=  self.skill_embed(skill)
        return p + e+sk+r+c

class DecoderEmbedding(nn.Module):
    def __init__(self, q_num, time_spend,  length,d_model):
        super(DecoderEmbedding, self).__init__()
        self.seq_len = length
        self.exercise_embed = nn.Embedding(q_num, d_model)
        self.category_embed = nn.Embedding(time_spend, d_model)
        self.position_embed = nn.Embedding(length, d_model)
        self.response_embed = nn.Embedding(2, d_model)
        self.skill_embed = nn.Embedding(40, d_model)

    def forward(self, exercises, categories,response,skill):
        e = self.exercise_embed(exercises)
        seq = torch.arange(self.seq_len).cuda().unsqueeze(0)
        p = self.position_embed(seq)
        sk=  self.skill_embed(skill)
        return e+sk+p


def distance_attn(source,num_heads):
    source=source.unsqueeze(-1)
    source=source.transpose(0, 1)
    src_len, bsz, embed_dim = source.size()
    ones=torch.ones(1,num_heads).cuda()
    source=source*ones
    q=source.contiguous().view(src_len, bsz * num_heads, 1).transpose(0, 1)
    k = source.contiguous().view(-1, bsz*num_heads, 1).transpose(0,1)
    attn_output_weights = q-k.transpose(1, 2)
    return attn_output_weights

class forgetting(nn.Module):
    def __init__(self):
        super(forgetting, self).__init__()
        self.a = nn.Parameter(torch.FloatTensor([0.1]))
        self.b = nn.Parameter(torch.FloatTensor([1]))
        self.c = nn.Parameter(torch.FloatTensor([1]))
        self.d = nn.Parameter(torch.FloatTensor([1]))

    def forward(self,time_done,num_heads):
        ttt=distance_attn(time_done,num_heads)
        return self.c*(self.b/(ttt+self.a))+self.d

def attn_time(time_done,num_heads):
    day_seven=1*24*60*60
    ttt=distance_attn(time_done,num_heads)
    ttt[ttt<day_seven]=0
    ttt[ttt>day_seven]=1
    return ttt.to(dtype=torch.bool)
    
def attn_skill(related_skill,num_heads):
    ttt=distance_attn(related_skill,num_heads)
    ttt[ttt==0]=0
    ttt[ttt!=0]=1
    return ttt.to(dtype=torch.bool)

class Learning(nn.Module):
    def __init__(self):
        super(Learning, self).__init__()
        self.linear_repeat= nn.Linear(1, 1)
        self.linear_last= nn.Linear(1, 1)
        self.PReLU_r = nn.PReLU()
        self.sig = nn.Sigmoid()
        self.PReLU_l = nn.PReLU()
        self.PReLU_a = nn.PReLU()
        self.ReLU = nn.ReLU()
        self.a = nn.Parameter(torch.FloatTensor([100]))
        self.c = nn.Parameter(torch.FloatTensor([0.1]))
        self.b = nn.Parameter(torch.FloatTensor([0.2]))
        self.d = nn.Parameter(torch.FloatTensor([-0.5]))
        self.linear_o= nn.Linear(1, 1)

    def forward(self,output, Last, Repeat):
        Repeat=self.ReLU(self.linear_repeat(Repeat.unsqueeze(-1).to(torch.float32)))     
        Last=self.ReLU(self.linear_last(Last.unsqueeze(-1).to(torch.float32)))
        learning_weight=self.sig(self.a*(Repeat/(Last+1)))
        learning_weight = learning_weight.permute(1, 0, 2)
        output=output+self.b*(learning_weight+self.d)*output*2
        return output


class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim 
        self.head_dim = embed_dim // num_heads  
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads  
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim
        self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))  
        self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim)) 
        self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))  
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.forget_module=forgetting()
        self._reset_parameters()

    def _reset_parameters(self):

        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)

    def forward(self, query, key, value, Time_done,mo,attn_mask=None, key_padding_mask=None):
        num_heads= self.num_heads
        dropout_p=self.dropout
        out_proj_weight=self.out_proj.weight
        out_proj_bias=self.out_proj.bias
        training=self.training
        q_proj_weight=self.q_proj_weight
        k_proj_weight=self.k_proj_weight
        v_proj_weight=self.v_proj_weight

        q = F.linear(query, q_proj_weight)
        k = F.linear(key, k_proj_weight)
        v = F.linear(value, v_proj_weight)
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads  
        scaling = float(head_dim) ** -0.5
        q = q * scaling  

        if attn_mask is not None: 
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0) 
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
                
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        if mo=='decoder':
            Forget=self.forget_module(time_done=Time_done,num_heads=num_heads)
            attn_output_weights=attn_output_weights*Forget

        if attn_mask is not None:
            attn_output_weights=attn_output_weights.masked_fill(attn_mask,float('-inf'))
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  
                float('-inf')) 
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,
                                                        src_len)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)  
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        Z = F.linear(attn_output, out_proj_weight, out_proj_bias)
        return Z, attn_output_weights.sum(dim=1) / num_heads 


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, length,dim_feedforward=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask, src_key_padding_mask,Time_done):

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,Time_done=Time_done,mo='encoder' )[0] 
        src = src + self.dropout1(src2) 
        src = self.norm1(src)  
        src2 = self.activation(self.linear1(src))  
        src2 = self.linear2(self.dropout(src2))  
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src 


class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers) 
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask, src_key_padding_mask,Time_done):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask,Time_done=Time_done)
        if self.norm is not None:
            output = self.norm(output)
        return output  

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()
        self.self_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.multihead_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu
        self.Learning=Learning()

    def forward(self, tgt, memory,Time_done, Last,Repeat,tgt_att_mask=None, memory_att_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt,  
                              attn_mask=tgt_att_mask,
                              key_padding_mask=tgt_key_padding_mask,Time_done=Time_done,mo='encoder')[0]
        tgt = tgt + self.dropout1(tgt2)  
        tgt = self.norm1(tgt) 
        tgt2 = self.multihead_attn(tgt, tgt, memory, 
                                   attn_mask=memory_att_mask,
                                   key_padding_mask=memory_key_padding_mask,Time_done=Time_done,mo='decoder')[0]
        tgt2=self.Learning(tgt2,Last,Repeat)
        tgt = tgt + self.dropout2(tgt2)  
        tgt = self.norm2(tgt)  
        tgt2 = self.activation(self.linear1(tgt)) 
        tgt2 = self.linear2(self.dropout(tgt2)) 
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt  

def misplace_add(memory,output):
    length,batch,dim=output.size()
    output=output[1:,:,:]
    output=torch.cat([output,torch.zeros(1,batch,dim).cuda()],dim=0)
    memory=memory+output
    return memory

class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,Time_done, Last,Repeat,tgt_att_mask=None, memory_att_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt 
        for mod in self.layers: 
            output = mod(output, memory,
                         tgt_att_mask=tgt_att_mask,
                         memory_att_mask=memory_att_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask
                         ,Time_done=Time_done
                         ,Last=Last,Repeat=Repeat
                         )
            memory=misplace_add(memory,output)
        if self.norm is not None:
            output = self.norm(output)
        return output  

class Model_exp(nn.Module):
    def __init__(self, q_num, time_spend,d_model, length,nhead,num_encoder_layers, dropout,speed_cate):
        super(Model_exp, self).__init__()

  #  ================ embedding =====================
        self.encoder_embedding = EncoderEmbedding(q_num=q_num,
                                                  time_spend=int(speed_cate),
                                                  length=length, d_model=d_model)
  #  ================ encoder =====================
        encoder_layer = MyTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout,length=length)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = MyTransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)
        self.att_mask = torch.triu(torch.ones(length, length),diagonal=1).to(dtype=torch.bool)
        ssa=torch.triu(torch.ones(length, length),diagonal=1)+torch.eye(length,length)
        ssa[0,0]=0
        self.att_mask_all=ssa.to(dtype=torch.bool)
  #  ================ decoder =====================
        self.decoder_embedding = DecoderEmbedding(q_num=q_num,
                                                    time_spend=time_spend,
                                                  length=length, d_model=d_model) 
        decoder_layer = MyTransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MyTransformerDecoder(decoder_layer=decoder_layer,num_layers= num_encoder_layers, norm=decoder_norm)
        self.fc_q = nn.Linear(d_model,q_num)
        self.q_onehot = torch.eye(q_num).cuda()
        self.dropout_fc=dropout

        self.out_fc = nn.Sequential(
            nn.Linear(512*2,256), nn.ReLU(),
            nn.Linear(256,40)
        )
        self.out_fc_2 = nn.Sequential(
            nn.Linear(40, 1),nn.Sigmoid()
        )
        self.out_time = nn.Sequential(
            nn.Linear(512,128), nn.ReLU(), nn.Dropout(self.dropout_fc),
            nn.Linear(128, int(speed_cate))
        )
        self.time_fc=nn.Linear(int(speed_cate),512)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.norm_layers = nn.LayerNorm(d_model) 
        self.s_onehot = torch.eye(172).cuda()
        self.sig = nn.Sigmoid()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    def forward(self,Q, Y, Skill,Spend, Last,Repeat, Done,S):
        enc = self.encoder_embedding(Q, Spend,Y,Skill)
        padding_mask=S==0
        enc=self.norm_layers(enc)
        encoder_output = self.encoder(src=enc.permute( 1, 0, 2),                            
                                    mask=self.att_mask.cuda()
                                    ,src_key_padding_mask=padding_mask.cuda(),Time_done=Done)

        dec = self.decoder_embedding(Q, Spend,Y,Skill)
        output = self.decoder(tgt=dec.permute( 1, 0, 2), memory=encoder_output, 
            tgt_att_mask=self.att_mask.cuda(), memory_att_mask=self.att_mask_all.cuda(),
                              tgt_key_padding_mask=padding_mask.cuda(),
                              memory_key_padding_mask=padding_mask.cuda(),
                              Time_done=Done
                              ,Last=Last,Repeat=Repeat)
        output = output.permute(1, 0, 2)
        out=output
        out_time = self.out_time(output)
        out_time, out, Y, S ,Spend= out_time[:, 1:], out[:, 1:], Y[:, 1:], S[:, 1:],Spend[:, 1:]
        ooo=self.time_fc(out_time)
        out = torch.cat([out, ooo], dim=-1)
        out = self.out_fc(out)
        out2=self.sig(out)
        out = self.out_fc_2(out)
        Z=out.squeeze(-1)
        return Z,Y,S,out_time,Spend,out2