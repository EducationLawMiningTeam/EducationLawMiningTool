import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将NumPy数组转换为PyTorch张量
a = np.load("data_rf_uc/hoi_feats.npy")
b = np.load("data_rf_uc/hoi_objs.npy")

a_tensor = torch.tensor(a, dtype=torch.float32).to(device)
b_tensor = torch.tensor(b, dtype=torch.long).to(device)


# 创建数据集和数据加载器
dataset = TensorDataset(a_tensor, b_tensor)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# 这段代码是为了初步测试 有没有分类unseen类obj的能力、



# 定义全连接分类器模型
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型和损失函数
model = Classifier(input_size=2048, num_classes=80).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    for inputs, labels in dataloader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss = total_loss + loss
        
        # 计算准确率
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    accuracy = total_correct / total_samples
    _loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {_loss.item():.4f}, Accuracy: {accuracy:.4f}')


# 保存模型
torch.save(model.state_dict(), 'data_rf_uc/classifier_model.pth')
