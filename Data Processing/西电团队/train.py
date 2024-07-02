import torch
import torch.nn as nn
from torch import  optim
from Data.ModelDataset_provider import FFTModelDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.engagement import Engagement
import os

def for_class(real_values):
    value_if_0 = torch.zeros(real_values.shape[0], dtype=torch.long).to(0)
    condition_0 = real_values <=0.8
    result = torch.where(condition_0,value_if_0,real_values)
    value_if_1 = torch.ones(real_values.shape[0], dtype=torch.long).to(0)
    condition_1 = (real_values >0.8) & (real_values <=1.2)
    result = torch.where(condition_1,value_if_1,result)
    value_if_2 = torch.full((real_values.shape[0],), 2, dtype=torch.long).to(0)
    condition_2 = real_values > 1.2 
    result = torch.where(condition_2,value_if_2,result)
    return result
def calculate_accuracy(true_labels, predicted_labels):
    correct = (true_labels == predicted_labels).sum()
    accuracy = correct 
    return accuracy
def Visualization(tensor_np):
    plt.figure(figsize=(10, 5))
    plt.plot(tensor_np, marker='o', linestyle='-')
    plt.title('Visualization ')
    plt.xlabel('Time')
    plt.ylabel('Engagement_Value')
    plt.grid(True)
    plt.show()
def _select_optimizer(model,learning_rate,):
    model_optim = optim.Adam(model.parameters(), lr=learning_rate)
    return model_optim
learning_rate = 5e-5
num_hidden_units = 23
class train_config_:
    d_model = 23 
    seq_len = 30
    pred_len = 3
    n_heads = 8
    label_len = 24
    moving_avg= 25
    num_epochs = 10
    output_attention = False
    freq = 's'
    d_layers = 2 
    e_layers = 2
    dropout=0.2  
    top_k = 5
    d_ff = 20 
    num_kernels = 3
    enc_in = 23  
    dec_in = 23 
    c_out = 23  
    activation = 'tanh'
    embed = "timeF"
    model_name = 'fed'#可选 fed、tcn
#fed为使用FEDformer做时序分析，tcn为使用tcn做时序分析
loss_function_classifi = nn.CrossEntropyLoss(torch.tensor([17/2.5,17/10,17/4.5], dtype=torch.float).to(0))
model = Engagement(train_config_)
optimizer = _select_optimizer(model,learning_rate)

files = [file for file in os.listdir("./dataset/feature/")]
file_list = []
for file in files:
    file_list.append(file)
all_count = 0
all_accuracy = 0
for epoch in range(50):
    model.train()
    for i in file_list:
        total_count = 0
        total_accuracy = 0.0
        file_name = ['./dataset/feature/'+i,'None','./dataset/engagement_value/'+i,]
        train_dataset = FFTModelDataset(file_name,0)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False,drop_last=True)
        result = torch.tensor([0.0]).to(0)
        for j,(x,y,yy,xx) in enumerate( train_loader ):
            optimizer.zero_grad()
            outputs = model(x)
            y = for_class(y)
            y = y.long()
            result = torch.cat((result,y),dim=0)
            loss = loss_function_classifi(outputs,y)
            predicted_labels = outputs.argmax(dim=-1)
            accuracy = calculate_accuracy(y, predicted_labels)
            total_accuracy += accuracy
            total_count += y.size(0)
            if(y.size(0) == 0):
                total_count +=1
            all_count += y.size(0)
            all_accuracy += accuracy
            loss.backward()
            optimizer.step()
        if(total_count ==0 ):
            print(i,epoch,total_count)
            continue
        else:
            tensor_np = result.cpu().numpy()
            Visualization(tensor_np)
            print(i+"训练结束,准确率为：",total_accuracy/total_count)

