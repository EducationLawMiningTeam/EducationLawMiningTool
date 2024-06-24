# 基本是参照train_unseen_classifier.py，主要是测试一下unseen能力

# 这个代码最好之后把labels改为actions，以免发生歧义

from hoi_arguments import parse_args
opt = parse_args()


from cls_models import ClsUnseenTrain
from generate import load_seen_att
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from mmdetection.splits import get_seen_class_ids
from numpy import linalg as LA
from util import *
from index2some import ua_object2valid_action, nf_uc_object2valid_action, rf_uc_object2valid_action, uo_object2valid_action, unsenn_nf_uc_object2action, unsenn_ua_object2action, unsenn_rf_uc_object2action, unsenn_uo_object2action

ua = [2, 10, 14, 20, 27, 33, 36, 42, 46, 57, 68, 81, 82, 86, 90, 92, 101, 103,
            109, 111, 116, 120, 121, 122, 123, 136, 137, 138, 140, 141, 149, 152, 155,
            160, 161, 170, 172, 174, 180, 188, 205, 208, 215, 222, 225, 236, 247, 260,
            265, 271, 273, 279, 283, 288, 295, 300, 301, 306, 310, 311, 315, 318, 319,
            337, 344, 352, 356, 363, 369, 373, 374, 419, 425, 427, 438, 453, 458, 461,
            464, 468, 471, 475, 480, 486, 489, 490, 496, 504, 506, 513, 516, 524, 528,
            533, 542, 555, 565, 576, 590, 597]

uv = [4, 6, 12, 15, 18, 25, 34, 38, 40, 49, 58, 60, 68, 69, 72, 73, 77, 82, 96, 97, 104, 113, 116, 118,
            122, 129, 139, 147,
            150, 153, 165, 166, 172, 175, 176, 181, 190, 202, 210, 212, 219, 227, 228, 233, 235, 243, 298, 313,
            315, 320, 326, 336,
            342, 345, 354, 372, 401, 404, 409, 431, 436, 459, 466, 470, 472, 479, 481, 488, 491, 494, 498, 504,
            519, 523, 535, 536,
            541, 544, 562, 565, 569, 572, 591, 595]

rf_uc = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                   70, 416,
                   389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596,
                   345, 189,
                   205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                   158, 195,
                   238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188,
                   216, 597,
                   77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104,
                   55, 50,
                   198, 168, 391, 192, 595, 136, 581]

nf_uc = [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
            212, 472, 61,
            457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479,
            230, 385, 73,
            159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
            29, 594, 346,
            456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191,
            266, 304, 6, 572,
            529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
            246, 173, 506,
            383, 93, 516, 64]

uo = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                    294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                    338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                    429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                    537, 558, 559, 560, 561, 595, 596, 597, 598, 599]



if opt.zero_shot_type == 'ua':
    unseen = ua
elif opt.zero_shot_type == 'uv':
    unseen = uv
elif opt.zero_shot_type == 'rf_uc':
    unseen = rf_uc
elif opt.zero_shot_type == 'nf_uc':
    unseen = nf_uc
elif opt.zero_shot_type == 'uo':
    unseen = uo


seen = [i for i in range(600)]
for i in unseen:
    seen.remove(i)


if opt.zero_shot_type == 'ua':
    limited_object2valid_action = ua_object2valid_action
if opt.zero_shot_type == 'nf_uc':
    limited_object2valid_action = nf_uc_object2valid_action
if opt.zero_shot_type == 'rf_uc':
    limited_object2valid_action = rf_uc_object2valid_action
if opt.zero_shot_type == 'uo':
    limited_object2valid_action = uo_object2valid_action


if opt.zero_shot_type == 'ua':
    unseen_limited_object2valid_action = unsenn_ua_object2action
if opt.zero_shot_type == 'nf_uc':
    unseen_limited_object2valid_action = unsenn_nf_uc_object2action
if opt.zero_shot_type == 'rf_uc':
    unseen_limited_object2valid_action = unsenn_rf_uc_object2action
if opt.zero_shot_type == 'uo':
    unseen_limited_object2valid_action = unsenn_uo_object2action


# path to save the trained classifier best checkpoint
path = f'data_{opt.zero_shot_type}/unseen_Classifier.pth'


att = np.load(opt.action_embedding)
att/=LA.norm(att, ord=2)
att = torch.tensor(att)

unseen_classifier = ClsUnseenTrain(att)
unseen_classifier.cuda()
# unseen_classifier = loadUnseenWeights(f'data_{opt.zero_shot_type}/unseen_Classifier.pth', unseen_classifier)



from index2some import hoi2obj, hoi2action
features_all = np.load(f"show_data/nf_uc_feats_all.npy")
labels_all = np.load(f"show_data/nf_uc_labels_all.npy")


actions_all = np.zeros((labels_all.shape[0], 117), dtype=int)
for i, label in enumerate(labels_all):
    actions_all[i][hoi2action(label)] = 1

objs_all = np.zeros(labels_all.shape[0], dtype=int)
for i, label in enumerate(labels_all):
    objs_all[i] = hoi2obj(label)

idx_seen = np.where(np.isin(labels_all, seen))[0]
idx_unseen = np.where(np.isin(labels_all, unseen))[0]


seen_feats = features_all[idx_seen]
seen_labels = actions_all[idx_seen]
seen_obj = objs_all[idx_seen]


unseen_feats = features_all[idx_unseen]
unseen_labels = actions_all[idx_unseen]
unseen_obj = objs_all[idx_unseen]


class Featuresdataset(Dataset):
     
    def __init__(self, features, labels, objs):
        self.features = features
        self.labels = labels
        self.objs = objs

    def __getitem__(self, idx):
        batch_feature = self.features[idx]
        batch_label = self.labels[idx]
        batch_obj = self.objs[idx]

        return batch_feature, batch_label, batch_obj

    def __len__(self):
        return len(self.labels)






# dataset_train = Featuresdataset(train_feats, train_labels, train_obj)
# dataloader_train = DataLoader(dataset_train, batch_size=512, shuffle=True) 
# dataset_test = Featuresdataset(test_feats, test_labels, test_obj)
# dataloader_test = DataLoader(dataset_test, batch_size=2048, shuffle=True) 
    
seen_dataset = Featuresdataset(seen_feats, seen_labels, seen_obj)
seen_dataloader = DataLoader(seen_dataset, batch_size=2048, shuffle=True) 
unseen_dataset = Featuresdataset(unseen_feats, unseen_labels, unseen_obj)
unseen_dataloader = DataLoader(unseen_dataset, batch_size=2048, shuffle=True) 


from torch.optim.lr_scheduler import StepLR

criterion = nn.BCELoss()

optimizer = optim.SGD(unseen_classifier.parameters(), lr=1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)



min_val_loss = float("inf")


def val(dataloader, f):
    running_loss = 0.0
    global min_val_loss
    unseen_classifier.eval()
    for i, (inputs, labels, objs) in enumerate(dataloader, 0):
        inputs = inputs.to(torch.float32).cuda()
        labels = labels.to(torch.float32).cuda()

        logits = unseen_classifier(inputs)
        prior = [f(i) for i in objs] # ****
        prior = np.array(prior)
        prior = torch.tensor(prior, dtype=torch.float32).cuda()

        outputs = torch.mul(torch.sigmoid(logits), prior)

        loss = criterion(outputs, labels)

        running_loss += loss.item()
    
    print(f'running loss {running_loss / (i+1)}')



val(seen_dataloader, limited_object2valid_action)
val(unseen_dataloader, unseen_limited_object2valid_action)



'''
如果是随机生成的模型
running loss 0.04071058746841219
running loss 0.02043945115150475

running loss 0.04070135640601317
running loss 0.020921291192857232

如果是训练过的模型
running loss 0.012121571797049709
running loss 0.017880821464265267


running loss 0.011418582923296425
running loss 0.018385100519148316  (这说明是没有unseen分类能力)

'''
