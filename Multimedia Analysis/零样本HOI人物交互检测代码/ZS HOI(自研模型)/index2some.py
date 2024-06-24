import torch
import numpy as np

num_action = 117
object_dict = ['airplane', 'apple', 'backpack', 'banana', 'baseball_bat', 'baseball_glove', 'bear', 'bed', 'bench', 'bicycle', 
               'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell_phone', 'chair', 
               'clock', 'couch', 'cow', 'cup', 'dining_table', 'dog', 'donut', 'elephant', 'fire_hydrant', 'fork', 'frisbee', 'giraffe', 
               'hair_drier', 'handbag', 'horse', 'hot_dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 
               'oven', 'parking_meter', 'person', 'pizza', 'potted_plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 
               'skateboard', 'skis', 'snowboard', 'spoon', 'sports_ball', 'stop_sign', 'suitcase', 'surfboard', 'teddy_bear', 'tennis_racket', 'tie', 
               'toaster', 'toilet', 'toothbrush', 'traffic_light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine_glass', 'zebra']

# with open('/data01/liuchuan/spatially-conditioned-graphs/hico_list_obj.txt', 'r') as file:
#     for line in file:
#         line = line.strip()
#         if line:
#             object_id, object_name = line.split('  ', 1)
#             object_dict.append(object_name)



verb_dict = ['adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 'buy', 
            'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook', 'cut', 'cut_with', 
            'direct', 'drag', 'dribble', 'drink_with', 'drive', 'dry', 'eat', 'eat_at', 'exit', 
            'feed', 'fill', 'flip', 'flush', 'fly', 'greet', 'grind', 'groom', 'herd', 'hit', 
            'hold', 'hop_on', 'hose', 'hug', 'hunt', 'inspect', 'install', 'jump', 'kick', 
            'kiss', 'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light', 'load', 'lose', 'make', 
            'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay', 
            'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read', 
            'release', 'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 
            'sign', 'sip', 'sit_at', 'sit_on', 'slide', 'smell', 'spin', 'squeeze', 'stab', 'stand_on',
              'stand_under', 'stick', 'stir', 'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach', 
              'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 'walk', 'wash', 'watch', 
              'wave', 'wear', 'wield', 'zip']

# with open('/data01/liuchuan/spatially-conditioned-graphs/hico_list_vb.txt', 'r') as file:
#     for line in file:
#         line = line.strip()
#         if line:
#             verb_id, verb_name = line.split('  ', 1)
#             verb_dict.append(verb_name)




hoi_dict = [['board', 'airplane'], ['direct', 'airplane'], ['exit', 'airplane'], ['fly', 'airplane'], ['inspect', 'airplane'], 
            ['load', 'airplane'], ['ride', 'airplane'], ['sit_on', 'airplane'], ['wash', 'airplane'], ['no_interaction', 'airplane'], 
            ['carry', 'bicycle'], ['hold', 'bicycle'], ['inspect', 'bicycle'], ['jump', 'bicycle'], ['hop_on', 'bicycle'], 
            ['park', 'bicycle'], ['push', 'bicycle'], ['repair', 'bicycle'], ['ride', 'bicycle'], ['sit_on', 'bicycle'], ['straddle', 'bicycle'], 
            ['walk', 'bicycle'], ['wash', 'bicycle'], ['no_interaction', 'bicycle'], ['chase', 'bird'], ['feed', 'bird'], ['hold', 'bird'], 
            ['pet', 'bird'], ['release', 'bird'], ['watch', 'bird'], ['no_interaction', 'bird'], ['board', 'boat'], ['drive', 'boat'],
              ['exit', 'boat'], ['inspect', 'boat'], ['jump', 'boat'], ['launch', 'boat'], ['repair', 'boat'], ['ride', 'boat'], ['row', 'boat'], 
            ['sail', 'boat'], ['sit_on', 'boat'], ['stand_on', 'boat'], ['tie', 'boat'], ['wash', 'boat'], ['no_interaction', 'boat'], ['carry', 'bottle'], 
            ['drink_with', 'bottle'], ['hold', 'bottle'], ['inspect', 'bottle'], ['lick', 'bottle'], ['open', 'bottle'], ['pour', 'bottle'], 
            ['no_interaction', 'bottle'], ['board', 'bus'], ['direct', 'bus'], ['drive', 'bus'], ['exit', 'bus'], ['inspect', 'bus'], ['load', 'bus'], 
            ['ride', 'bus'], ['sit_on', 'bus'], ['wash', 'bus'], ['wave', 'bus'], ['no_interaction', 'bus'], ['board', 'car'], ['direct', 'car'], ['drive', 'car'], 
            ['hose', 'car'], ['inspect', 'car'], ['jump', 'car'], ['load', 'car'], ['park', 'car'], ['ride', 'car'], ['wash', 'car'], ['no_interaction', 'car'],
              ['dry', 'cat'], ['feed', 'cat'], ['hold', 'cat'], ['hug', 'cat'], ['kiss', 'cat'], ['pet', 'cat'], ['scratch', 'cat'], ['wash', 'cat'], 
              ['chase', 'cat'], ['no_interaction', 'cat'], ['carry', 'chair'], ['hold', 'chair'], ['lie_on', 'chair'], ['sit_on', 'chair'], ['stand_on', 'chair'], 
              ['no_interaction', 'chair'], ['carry', 'couch'], ['lie_on', 'couch'], ['sit_on', 'couch'], ['no_interaction', 'couch'], ['feed', 'cow'], 
            ['herd', 'cow'], ['hold', 'cow'], ['hug', 'cow'], ['kiss', 'cow'], ['lasso', 'cow'], ['milk', 'cow'], ['pet', 'cow'], ['ride', 'cow'], 
            ['walk', 'cow'], ['no_interaction', 'cow'], ['clean', 'dining_table'], ['eat_at', 'dining_table'], ['sit_at', 'dining_table'], 
            ['no_interaction', 'dining_table'], ['carry', 'dog'], ['dry', 'dog'], ['feed', 'dog'], ['groom', 'dog'], ['hold', 'dog'], ['hose', 'dog'], 
            ['hug', 'dog'], ['inspect', 'dog'], ['kiss', 'dog'], ['pet', 'dog'], ['run', 'dog'], ['scratch', 'dog'], ['straddle', 'dog'], ['train', 'dog'], 
            ['walk', 'dog'], ['wash', 'dog'], ['chase', 'dog'], ['no_interaction', 'dog'], ['feed', 'horse'], ['groom', 'horse'], ['hold', 'horse'], ['hug', 'horse'], 
            ['jump', 'horse'], ['kiss', 'horse'], ['load', 'horse'], ['hop_on', 'horse'], ['pet', 'horse'], ['race', 'horse'], ['ride', 'horse'], ['run', 'horse'], 
            ['straddle', 'horse'], ['train', 'horse'], ['walk', 'horse'], ['wash', 'horse'], ['no_interaction', 'horse'], ['hold', 'motorcycle'], 
            ['inspect', 'motorcycle'], ['jump', 'motorcycle'], ['hop_on', 'motorcycle'], ['park', 'motorcycle'], ['push', 'motorcycle'], ['race', 'motorcycle'],
              ['ride', 'motorcycle'], ['sit_on', 'motorcycle'], ['straddle', 'motorcycle'], ['turn', 'motorcycle'], ['walk', 'motorcycle'], ['wash', 'motorcycle'], 
              ['no_interaction', 'motorcycle'], ['carry', 'person'], ['greet', 'person'], ['hold', 'person'], ['hug', 'person'], ['kiss', 'person'], ['stab', 'person'],
            ['tag', 'person'], ['teach', 'person'], ['lick', 'person'], ['no_interaction', 'person'], ['carry', 'potted_plant'], ['hold', 'potted_plant'], 
            ['hose', 'potted_plant'], ['no_interaction', 'potted_plant'], ['carry', 'sheep'], ['feed', 'sheep'], ['herd', 'sheep'], ['hold', 'sheep'], 
            ['hug', 'sheep'], ['kiss', 'sheep'], ['pet', 'sheep'], ['ride', 'sheep'], ['shear', 'sheep'], ['walk', 'sheep'], ['wash', 'sheep'], 
            ['no_interaction', 'sheep'], ['board', 'train'], ['drive', 'train'], ['exit', 'train'], ['load', 'train'], ['ride', 'train'], ['sit_on', 'train'], 
            ['wash', 'train'], ['no_interaction', 'train'], ['control', 'tv'], ['repair', 'tv'], ['watch', 'tv'], ['no_interaction', 'tv'], ['buy', 'apple'], 
            ['cut', 'apple'], ['eat', 'apple'], ['hold', 'apple'], ['inspect', 'apple'], ['peel', 'apple'], ['pick', 'apple'], ['smell', 'apple'], ['wash', 'apple'], 
            ['no_interaction', 'apple'], ['carry', 'backpack'], ['hold', 'backpack'], ['inspect', 'backpack'], ['open', 'backpack'], ['wear', 'backpack'], 
            ['no_interaction', 'backpack'], ['buy', 'banana'], ['carry', 'banana'], ['cut', 'banana'], ['eat', 'banana'], ['hold', 'banana'], ['inspect', 'banana'], 
            ['peel', 'banana'], ['pick', 'banana'], ['smell', 'banana'], ['no_interaction', 'banana'], ['break', 'baseball_bat'], ['carry', 'baseball_bat'], 
            ['hold', 'baseball_bat'], ['sign', 'baseball_bat'], ['swing', 'baseball_bat'], ['throw', 'baseball_bat'], ['wield', 'baseball_bat'], 
            ['no_interaction', 'baseball_bat'], ['hold', 'baseball_glove'], ['wear', 'baseball_glove'], ['no_interaction', 'baseball_glove'], ['feed', 'bear'], 
            ['hunt', 'bear'], ['watch', 'bear'], ['no_interaction', 'bear'], ['clean', 'bed'], ['lie_on', 'bed'], ['sit_on', 'bed'], ['no_interaction', 'bed'], 
            ['inspect', 'bench'], ['lie_on', 'bench'], ['sit_on', 'bench'], ['no_interaction', 'bench'], ['carry', 'book'], ['hold', 'book'], ['open', 'book'], 
            ['read', 'book'], ['no_interaction', 'book'], ['hold', 'bowl'], ['stir', 'bowl'], ['wash', 'bowl'], ['lick', 'bowl'], ['no_interaction', 'bowl'], 
            ['cut', 'broccoli'], ['eat', 'broccoli'], ['hold', 'broccoli'], ['smell', 'broccoli'], ['stir', 'broccoli'], ['wash', 'broccoli'], 
            ['no_interaction', 'broccoli'], ['blow', 'cake'], ['carry', 'cake'], ['cut', 'cake'], ['eat', 'cake'], ['hold', 'cake'], ['light', 'cake'], 
            ['make', 'cake'], ['pick_up', 'cake'], ['no_interaction', 'cake'], ['carry', 'carrot'], ['cook', 'carrot'], ['cut', 'carrot'], ['eat', 'carrot'], 
            ['hold', 'carrot'], ['peel', 'carrot'], ['smell', 'carrot'], ['stir', 'carrot'], ['wash', 'carrot'], ['no_interaction', 'carrot'], ['carry', 'cell_phone'],
            ['hold', 'cell_phone'], ['read', 'cell_phone'], ['repair', 'cell_phone'], ['talk_on', 'cell_phone'], ['text_on', 'cell_phone'], 
            ['no_interaction', 'cell_phone'], ['check', 'clock'], ['hold', 'clock'], ['repair', 'clock'], ['set', 'clock'], ['no_interaction', 'clock'], 
            ['carry', 'cup'], ['drink_with', 'cup'], ['hold', 'cup'], ['inspect', 'cup'], ['pour', 'cup'], ['sip', 'cup'], ['smell', 'cup'], ['fill', 'cup'], 
            ['wash', 'cup'], ['no_interaction', 'cup'], ['buy', 'donut'], ['carry', 'donut'], ['eat', 'donut'], ['hold', 'donut'], ['make', 'donut'], 
            ['pick_up', 'donut'], ['smell', 'donut'], ['no_interaction', 'donut'], ['feed', 'elephant'], ['hold', 'elephant'], ['hose', 'elephant'], 
            ['hug', 'elephant'], ['kiss', 'elephant'], ['hop_on', 'elephant'], ['pet', 'elephant'], ['ride', 'elephant'], ['walk', 'elephant'], ['wash', 'elephant'], 
            ['watch', 'elephant'], ['no_interaction', 'elephant'], ['hug', 'fire_hydrant'], ['inspect', 'fire_hydrant'], ['open', 'fire_hydrant'], 
            ['paint', 'fire_hydrant'], ['no_interaction', 'fire_hydrant'], ['hold', 'fork'], ['lift', 'fork'], ['stick', 'fork'], ['lick', 'fork'], ['wash', 'fork'],
              ['no_interaction', 'fork'], ['block', 'frisbee'], ['catch', 'frisbee'], ['hold', 'frisbee'], ['spin', 'frisbee'], ['throw', 'frisbee'], 
              ['no_interaction', 'frisbee'], ['feed', 'giraffe'], ['kiss', 'giraffe'], ['pet', 'giraffe'], ['ride', 'giraffe'], ['watch', 'giraffe'], 
              ['no_interaction', 'giraffe'], ['hold', 'hair_drier'], ['operate', 'hair_drier'], ['repair', 'hair_drier'], ['no_interaction', 'hair_drier'], 
              ['carry', 'handbag'], ['hold', 'handbag'], ['inspect', 'handbag'], ['no_interaction', 'handbag'], ['carry', 'hot_dog'], ['cook', 'hot_dog'], 
              ['cut', 'hot_dog'], ['eat', 'hot_dog'], ['hold', 'hot_dog'], ['make', 'hot_dog'], ['no_interaction', 'hot_dog'], ['carry', 'keyboard'], 
              ['clean', 'keyboard'], ['hold', 'keyboard'], ['type_on', 'keyboard'], ['no_interaction', 'keyboard'], ['assemble', 'kite'], ['carry', 'kite'], 
              ['fly', 'kite'], ['hold', 'kite'], ['inspect', 'kite'], ['launch', 'kite'], ['pull', 'kite'], ['no_interaction', 'kite'], ['cut_with', 'knife'], 
              ['hold', 'knife'], ['stick', 'knife'], ['wash', 'knife'], ['wield', 'knife'], ['lick', 'knife'], ['no_interaction', 'knife'], ['hold', 'laptop'],
            ['open', 'laptop'], ['read', 'laptop'], ['repair', 'laptop'], ['type_on', 'laptop'], ['no_interaction', 'laptop'], ['clean', 'microwave'], ['open', 'microwave'], ['operate', 'microwave'], ['no_interaction', 'microwave'], ['control', 'mouse'], ['hold', 'mouse'], ['repair', 'mouse'], ['no_interaction', 'mouse'], ['buy', 'orange'], ['cut', 'orange'], ['eat', 'orange'], ['hold', 'orange'], ['inspect', 'orange'], ['peel', 'orange'], ['pick', 'orange'], ['squeeze', 'orange'], ['wash', 'orange'], ['no_interaction', 'orange'], ['clean', 'oven'], ['hold', 'oven'], ['inspect', 'oven'], ['open', 'oven'], ['repair', 'oven'], ['operate', 'oven'], ['no_interaction', 'oven'], ['check', 'parking_meter'], ['pay', 'parking_meter'], ['repair', 'parking_meter'], ['no_interaction', 'parking_meter'], ['buy', 'pizza'], ['carry', 'pizza'], ['cook', 'pizza'], ['cut', 'pizza'], ['eat', 'pizza'], ['hold', 'pizza'], ['make', 'pizza'], ['pick_up', 'pizza'], ['slide', 'pizza'], ['smell', 'pizza'], ['no_interaction', 'pizza'], ['clean', 'refrigerator'], ['hold', 'refrigerator'], ['move', 'refrigerator'], ['open', 'refrigerator'], ['no_interaction', 'refrigerator'], ['hold', 'remote'], ['point', 'remote'], ['swing', 'remote'], ['no_interaction', 'remote'], ['carry', 'sandwich'], ['cook', 'sandwich'], ['cut', 'sandwich'], ['eat', 'sandwich'], ['hold', 'sandwich'], ['make', 'sandwich'], ['no_interaction', 'sandwich'], ['cut_with', 'scissors'], ['hold', 'scissors'], ['open', 'scissors'], ['no_interaction', 'scissors'], ['clean', 'sink'], ['repair', 'sink'], ['wash', 'sink'], ['no_interaction', 'sink'], ['carry', 'skateboard'], ['flip', 'skateboard'], ['grind', 'skateboard'], ['hold', 'skateboard'], ['jump', 'skateboard'], ['pick_up', 'skateboard'], ['ride', 'skateboard'], ['sit_on', 'skateboard'], ['stand_on', 'skateboard'], ['no_interaction', 'skateboard'], ['adjust', 'skis'], ['carry', 'skis'], ['hold', 'skis'], ['inspect', 'skis'], ['jump', 'skis'], ['pick_up', 'skis'], ['repair', 'skis'], ['ride', 'skis'], ['stand_on', 'skis'], ['wear', 'skis'], ['no_interaction', 'skis'], ['adjust', 'snowboard'], ['carry', 'snowboard'], ['grind', 'snowboard'], ['hold', 'snowboard'], ['jump', 'snowboard'], ['ride', 'snowboard'], ['stand_on', 'snowboard'], ['wear', 'snowboard'], ['no_interaction', 'snowboard'], ['hold', 'spoon'], ['lick', 'spoon'], ['wash', 'spoon'], ['sip', 'spoon'], ['no_interaction', 'spoon'], ['block', 'sports_ball'], ['carry', 'sports_ball'], ['catch', 'sports_ball'], ['dribble', 'sports_ball'], ['hit', 'sports_ball'], ['hold', 'sports_ball'], ['inspect', 'sports_ball'], ['kick', 'sports_ball'], ['pick_up', 'sports_ball'], ['serve', 'sports_ball'], ['sign', 'sports_ball'], ['spin', 'sports_ball'], ['throw', 'sports_ball'], ['no_interaction', 'sports_ball'], ['hold', 'stop_sign'], ['stand_under', 'stop_sign'], ['stop_at', 'stop_sign'], ['no_interaction', 'stop_sign'], ['carry', 'suitcase'], ['drag', 'suitcase'], ['hold', 'suitcase'], ['hug', 'suitcase'], ['load', 'suitcase'], ['open', 'suitcase'], ['pack', 'suitcase'], ['pick_up', 'suitcase'], ['zip', 'suitcase'], ['no_interaction', 'suitcase'], ['carry', 'surfboard'], ['drag', 'surfboard'], ['hold', 'surfboard'], ['inspect', 'surfboard'], ['jump', 'surfboard'], ['lie_on', 'surfboard'], ['load', 'surfboard'], ['ride', 'surfboard'], ['stand_on', 'surfboard'], ['sit_on', 'surfboard'], ['wash', 'surfboard'], ['no_interaction', 'surfboard'], ['carry', 'teddy_bear'], ['hold', 'teddy_bear'], ['hug', 'teddy_bear'], ['kiss', 'teddy_bear'], ['no_interaction', 'teddy_bear'], ['carry', 'tennis_racket'], ['hold', 'tennis_racket'], ['inspect', 'tennis_racket'], ['swing', 'tennis_racket'], ['no_interaction', 'tennis_racket'], ['adjust', 'tie'], ['cut', 'tie'], ['hold', 'tie'], ['inspect', 'tie'], ['pull', 'tie'], ['tie', 'tie'], ['wear', 'tie'], ['no_interaction', 'tie'], ['hold', 'toaster'], ['operate', 'toaster'], ['repair', 'toaster'], ['no_interaction', 'toaster'], ['clean', 'toilet'], ['flush', 'toilet'], ['open', 'toilet'], ['repair', 'toilet'], ['sit_on', 'toilet'], ['stand_on', 'toilet'], ['wash', 'toilet'], ['no_interaction', 'toilet'], ['brush_with', 'toothbrush'], ['hold', 'toothbrush'], ['wash', 'toothbrush'], ['no_interaction', 'toothbrush'], ['install', 'traffic_light'], ['repair', 'traffic_light'], ['stand_under', 'traffic_light'], ['stop_at', 'traffic_light'], ['no_interaction', 'traffic_light'], ['direct', 'truck'], ['drive', 'truck'], ['inspect', 'truck'], ['load', 'truck'], ['repair', 'truck'], ['ride', 'truck'], ['sit_on', 'truck'], ['wash', 'truck'], ['no_interaction', 'truck'], ['carry', 'umbrella'], ['hold', 'umbrella'], ['lose', 'umbrella'], ['open', 'umbrella'], ['repair', 'umbrella'], ['set', 'umbrella'], ['stand_under', 'umbrella'], ['no_interaction', 'umbrella'], ['hold', 'vase'], ['make', 'vase'], ['paint', 'vase'], ['no_interaction', 'vase'], ['fill', 'wine_glass'], ['hold', 'wine_glass'], ['sip', 'wine_glass'], ['toast', 'wine_glass'], ['lick', 'wine_glass'], ['wash', 'wine_glass'], ['no_interaction', 'wine_glass'], ['feed', 'zebra'], ['hold', 'zebra'], ['pet', 'zebra'], ['watch', 'zebra'], ['no_interaction', 'zebra']]
# with open('/data01/liuchuan/spatially-conditioned-graphs/hico_list_hoi.txt', "r") as file:
#     lines = file.readlines()
#     for line in lines:
#         columns = line.split()
#         if len(columns) >= 3:
#             verb = columns[2]
#             obj = columns[1]
#             id = columns[0]
#             hoi_dict.append([verb, obj])


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

def objaction2hoi(obj:int, action:int):
    obj_name = object_dict[obj]
    action_name = verb_dict[action]
    return hoi_dict.index([action_name, obj_name])



object2action_2 = torch.load('data/object2action_2.pt')
def object2valid_action(obj:int):
    return object2action_2[obj]

ua_object2action_2 = torch.load('data_ua/ua_object2action_2.pt')
def ua_object2valid_action(obj:int):
    return ua_object2action_2[obj]

unsenn_ua_object2action_2 = torch.load('data_ua/unsenn_ua_object2action_2.pt')
def unsenn_ua_object2action(obj:int):
    return unsenn_ua_object2action_2[obj]


rf_uc_object2action_2 = torch.load('data_rf_uc/rf_uc_object2action_2.pt')
def rf_uc_object2valid_action(obj:int):
    return rf_uc_object2action_2[obj]

unsenn_rf_uc_object2action_2 = torch.load('data_rf_uc/unseen_rf_uc_object2action_2.pt')
def unsenn_rf_uc_object2action(obj:int):
    return unsenn_rf_uc_object2action_2[obj]

nf_uc_object2action_2 = torch.load('data_nf_uc/nf_uc_object2action_2.pt')
def nf_uc_object2valid_action(obj:int):
    return nf_uc_object2action_2[obj]

unsenn_nf_uc_object2action_2 = torch.load('data_nf_uc/unseen_nf_uc_object2action_2.pt')
def unsenn_nf_uc_object2action(obj:int):
    return unsenn_nf_uc_object2action_2[obj]


uo_object2action_2 = torch.load('data_uo/uo_object2action_2.pt')
def uo_object2valid_action(obj:int):
    return uo_object2action_2[obj]

unsenn_uo_object2action_2 = torch.load('data_uo/unseen_uo_object2action_2.pt')
def unsenn_uo_object2action(obj:int):
    return unsenn_uo_object2action_2[obj]


uv_object2action_2 = torch.load('data_uv/uv_object2action_2.pt')
def uv_object2valid_action(obj:int):
    return uv_object2action_2[obj]

unsenn_uv_object2action_2 = torch.load('data_uv/unseen_uv_object2action_2.pt')
def unsenn_uv_object2action(obj:int):
    return unsenn_uv_object2action_2[obj]