import pickle
import json
import numpy as np
# 指定.pkl文件的路径

# file_path = "Test_HICO_cascade_rcnn_X152_FPN_lr1e-3_20k.pkl"
file_path = "Test_HICO_res101_3x_FPN_hico.pkl"
# file_path = "Test_HICO_detr-r101-hicodet.pkl"
file_path = "Test_HICO_detr-r50-hicodet.pkl"

# 使用pickle模块加载.pkl文件
with open(file_path, "rb") as file:
    loaded_data = pickle.load(file)


# coco标注到hico标注的转换
    
with open('hicodet/coco80tohico80.json', 'r') as file:
    coco2hico = json.load(file)

print(coco2hico)

label1 = []
label2 = []

for i in loaded_data:
    boxes = []
    labels = []
    scores = []

    for instances in loaded_data[i]:

        if instances[5] >= 0.001:
            boxes.append(instances[2].tolist())
            labels.append( coco2hico[ str(instances[4]) ]) 

            label1.append( coco2hico[ str(instances[4]) ] )
            label2.append( instances[4] )

            scores.append(instances[5].tolist())

    


    data = {
        "boxes": boxes,
        "labels": labels,
        "scores":scores,
    }

    
    # with open('hicodet/detections/VCL/HICO_test2015_{:08}.json'.format(i), 'w') as file:
    #     json.dump(data, file)

print(set(label1))
print(set(label2))
object_dict = []

with open('/data01/liuchuan/spatially-conditioned-graphs/hico_list_obj.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            object_id, object_name = line.split('  ', 1)
            object_dict.append(object_name)

for i in range (80):
    if i not in set(label1):
        print(object_dict[i])


import torch
model = torch.load('model_0064999.pth')

pass