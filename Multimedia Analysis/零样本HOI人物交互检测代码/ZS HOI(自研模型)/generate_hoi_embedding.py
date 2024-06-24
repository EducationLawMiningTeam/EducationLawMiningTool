import clip
import torch
import numpy as np

file_path = "hico_list_hoi.txt"  # 替换为您的文件路径
device = "cuda" if torch.cuda.is_available() else "cpu"
result_list = []

with open(file_path, "r") as file:
    lines = file.readlines()
    for line in lines:
        columns = line.split()
        if len(columns) >= 3:
            verb = columns[2]
            obj = columns[1]
            result_list.append(f"a person {verb} {obj}")

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32")

text = clip.tokenize(result_list)
text = text.to(device)  # 如果有GPU可用，将输入移至GPU


# 使用CLIP模型编码文本
with torch.no_grad():
    text_features = model.encode_text(text)

text_features = text_features.cpu().numpy()
text_features = text_features.astype(np.float32)

np.save('hoi_embedding_2.npy', text_features)

print(text_features)
