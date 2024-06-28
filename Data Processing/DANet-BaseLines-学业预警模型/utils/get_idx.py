import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据集
file_path = '../data/student_behavior/DataSet_V5_filled.xlsx'
data = pd.read_excel(file_path)

# 划分数据集：首先划分训练集和临时集(验证集+测试集)
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)

# 从临时集中划分出验证集和测试集
valid_data, test_data = train_test_split(temp_data, test_size=0.3, random_state=42)

# 保存索引到CSV文件
train_idx = train_data.index
valid_idx = valid_data.index
test_idx = test_data.index

train_idx.to_series().to_csv('../data/student_behavior/train_idx.csv', index=False)
valid_idx.to_series().to_csv('../data/student_behavior/valid_idx.csv', index=False)
test_idx.to_series().to_csv('../data/student_behavior/test_idx.csv', index=False)

# 加载并检查验证集的索引
valid_idx_loaded = pd.read_csv('../data/student_behavior/valid_idx.csv')['0'].values
print(valid_idx_loaded)
