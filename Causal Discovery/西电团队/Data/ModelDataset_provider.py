import math
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from scipy.fft import rfft
import numpy as np
from scipy.fft import irfft
def trenddetch(y,threadhold = 3):
    trendlist = []
    yf   = rfft(np.array(y))
    yf_abs = np.abs(yf)
    for i in range(2, 11):
        max_values = sorted(yf_abs, reverse=True)[:i+1]
        indices = yf_abs > max_values[-1]  # filter out those value under 300
        yf_clean = indices * yf  # noise frequency will be set to 0
        new_f_clean = irfft(yf_clean)
        trendlist.append(new_f_clean)
    return trendlist[6]
class FFTModelDataset(Dataset,):
    def __init__(self,data_path,device,seq_len = 60,pre_len = 3,threadhold = 3):
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.data_path = data_path
        self.removehead  = 0
        self.data_feature = pd.read_csv(data_path[0])
        self.data_engagement_value = pd.read_csv(data_path[2])
        self.data_pose_temp  = pd.DataFrame([self.data_feature[' pose_Tx'],self.data_feature[' pose_Ty'],self.data_feature[' pose_Tz'],
                                             self.data_feature[' pose_Rx'], self.data_feature[' pose_Ry'],
                                             self.data_feature[' pose_Rz']]).T[:math.floor(self.data_feature.shape[0]/60)*60]
        self.data_face_temp = pd.DataFrame([self.data_feature[' AU01_r'],self.data_feature[' AU02_r'],self.data_feature[' AU04_r'],
                                            self.data_feature[' AU05_r'],self.data_feature[' AU06_r'],self.data_feature[' AU07_r'],
                                            self.data_feature[' AU09_r'],self.data_feature[' AU10_r'],self.data_feature[' AU12_r'],
                                            self.data_feature[' AU14_r'],self.data_feature[' AU15_r'],self.data_feature[' AU17_r'],
                                            self.data_feature[' AU20_r'],self.data_feature[' AU23_r'],self.data_feature[' AU25_r'],
                                            self.data_feature[' AU26_r'],
                                            self.data_feature[' AU45_r']]).T[:math.floor(self.data_feature.shape[0]/60)*60]
        self.data_face_temp_c = pd.DataFrame([self.data_feature[' AU01_c'],self.data_feature[' AU02_c'],self.data_feature[' AU04_c'],
                                            self.data_feature[' AU05_c'],self.data_feature[' AU06_c'],self.data_feature[' AU07_c'],
                                            self.data_feature[' AU09_c'],self.data_feature[' AU10_c'],self.data_feature[' AU12_c'],
                                            self.data_feature[' AU14_c'],self.data_feature[' AU15_c'],self.data_feature[' AU17_c'],
                                            self.data_feature[' AU20_c'],self.data_feature[' AU23_c'],self.data_feature[' AU25_c'],
                                            self.data_feature[' AU26_c'],
                                            self.data_feature[' AU45_c']]).T[:math.floor(self.data_feature.shape[0]/60)*60]
        self.int_index= math.floor(self.data_feature.shape[0]/60)
        self.data_pose_temp = self.data_pose_temp.groupby(self.data_pose_temp.index//60).mean()
        self.data_face_temp = self.data_face_temp.groupby(self.data_face_temp.index//60).mean()
        self.data_face_temp_c = self.data_face_temp_c.groupby(self.data_face_temp_c.index//60).mean()
        self.data_contact = pd.concat([self.data_pose_temp,self.data_face_temp_c],axis=1)
        if(self.data_contact.shape[0]>self.data_engagement_value.shape[0]):
            tail = self.data_engagement_value.shape[0]
            self.data_contact = self.data_contact[self.data_contact.shape[0]-self.data_engagement_value.shape[0]:]
        else:
            tail = self.data_contact.shape[0]
            self.data_engagement_value = self.data_engagement_value[self.data_engagement_value.shape[0]-self.data_contact.shape[0]:]
        self.data_contact = self.data_contact[8+self.removehead:tail]
        self.data_engagement_value_ = pd.DataFrame(self.data_engagement_value[8+self.removehead:tail].astype(np.float32)[list(self.data_engagement_value.columns)[-1]])
        self.data_engagement_value_ = self.data_engagement_value_.reset_index(drop = True)
        self.data_contact = self.data_contact.reset_index(drop = True)
        self.data_contact['group'] = self.data_contact.index //3
        self.data_contact = self.data_contact.groupby(['group']).mean()
        self.data_engagement_value_['group'] = self.data_engagement_value_.index //3
        self.data_engagement_value_ = self.data_engagement_value_.groupby(['group'])
        self.data_engagement_value = self.data_engagement_value[8+self.removehead:tail].astype(
            float)[list(self.data_engagement_value.columns)[-1]]
        self.y = torch.tensor(self.data_engagement_value.values,dtype=torch.float32).to(device)
        self.X  = torch.tensor(self.data_contact.values,dtype=torch.float32).to(device)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,index):
        if index >= self.seq_len :
            i_start = index - self.seq_len
            x = self.X[i_start:(i_start+self.seq_len), :]
        else:
            padding = self.X[0].repeat(self.seq_len - index-1 ,1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[index],self.y[index:index+self.pre_len],self.X[index:index+self.pre_len,:]



