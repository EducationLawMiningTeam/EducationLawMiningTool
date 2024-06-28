import pandas as pd
from imblearn.over_sampling import SMOTE

# 读取原始数据
file_path = '../data/student_behavior/DataSet_V5_filled.xlsx'
data = pd.read_excel(file_path)
data = data.drop(columns=['xh'])
# 假设最后一列是标签列，其余列是特征
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 使用SMOTE算法生成少样本数据
sampling_strategy = {1: 300, 2: 300, 3: 300, 4: 300}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 将生成的新样本数据与原始数据合并
data_resampled = pd.concat([X_res, y_res], axis=1)

# 保存合并后的数据到新的Excel文件
output_file_path = '../data/student_behavior/DataSet_V5_smote.xlsx'
data_resampled.to_excel(output_file_path, index=False)
