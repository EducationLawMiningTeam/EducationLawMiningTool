import torch
import numpy as np

num_action = 117
object_dict = []

with open('/data01/liuchuan/spatially-conditioned-graphs/hico_list_obj.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            object_id, object_name = line.split('  ', 1)
            object_dict.append(object_name)

verb_dict = []

with open('/data01/liuchuan/spatially-conditioned-graphs/hico_list_vb.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            verb_id, verb_name = line.split('  ', 1)
            verb_dict.append(verb_name)

hoi_dict = []
with open('/data01/liuchuan/spatially-conditioned-graphs/hico_list_hoi.txt', "r") as file:
    lines = file.readlines()
    for line in lines:
        columns = line.split()
        if len(columns) >= 3:
            verb = columns[2]
            obj = columns[1]
            id = columns[0]
            hoi_dict.append([verb, obj])

with open('/data01/liuchuan/spatially-conditioned-graphs/hoi_dict.txt', 'w') as file:
    file.write(str(hoi_dict))


def hoiindex2hoi(index:int):
    return hoi_dict[index]

def hoi2hoiindex(hoi):
    return hoi_dict.index(hoi)

def hoi2actionv2(hoi:str):
    act, obj = hoi.split(',')
    return verb_dict.index(act)

def hoi2action(hoi:int):
    return verb_dict.index(hoi_dict[hoi][0])

def hoi2obj(hoi:int):
    return object_dict.index(hoi_dict[hoi][1])

def hoi2objv2(hoi:str):
    act, obj = hoi.split(',')
    return object_dict.index(obj)


object2action_2 = torch.load('/data01/liuchuan/zero_shot_detection-master/data/object2action_2.pt')
def object2valid_action(obj:int):
    return object2action_2[obj]
