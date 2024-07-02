import torch
import torch.nn as nn
from .FEDformer import Model
from .TCN import TemporalConvNet
import numpy as np
class Engagement(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.configs = configs
        self.num_hidden_units = 23
        "使用fedformer"
        self.fedmodel = Model(self.configs).to(0)
        self.linear_for_fed = nn.Linear(self.num_hidden_units, 3).to(0)
        "tcn model"
        self.finaldin  = 8
        self.tcn   = TemporalConvNet(self.num_hidden_units,[64,32,16,self.finaldin]).to(0)
        self.linear_for_tcn = nn.Linear(self.finaldin, 3).to(0)
    def forward(self,x):
        if(self.configs.model_name == 'fed'):
            forecast = self.fedmodel(x)
            output = self.linear_for_fed(forecast)
            output = output[:,0,:]
            output = torch.softmax(output, dim=-1)
            return output
        if(self.configs.model_name == 'tcn'):
            x= x.transpose(1,2)
            forecast = self.tcn(x)
            output = self.linear_for_tcn(forecast[:, :, -1])
            output = torch.softmax(output, dim=-1)
            return output

