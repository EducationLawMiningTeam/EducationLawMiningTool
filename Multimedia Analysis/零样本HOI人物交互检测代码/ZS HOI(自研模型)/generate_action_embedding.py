import clip
import torch
import numpy as np

file_path = "/data01/liuchuan/spatially-conditioned-graphs/hico_list_vb.txt"  # 替换为您的文件路径
device = "cuda" if torch.cuda.is_available() else "cpu"
result_list = []

with open(file_path, "r") as file:
    lines = file.readlines()
    for line in lines:
        columns = line.split()
        if len(columns) >= 2:
            verb = columns[1]

            result_list.append(verb)

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32")

text = clip.tokenize(result_list)
text = text.to(device)  # 如果有GPU可用，将输入移至GPU


# 使用CLIP模型编码文本
with torch.no_grad():
    text_features = model.encode_text(text)

text_features = text_features.cpu().numpy()
text_features = text_features.astype(np.float32)

l2_norm = np.linalg.norm(text_features, axis=1)
print(l2_norm)
np.save('hoi_action_embedding.npy', text_features)

print(text_features)
