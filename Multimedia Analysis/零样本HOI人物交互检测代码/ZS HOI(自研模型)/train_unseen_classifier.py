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
from index2some import ua_object2valid_action, nf_uc_object2valid_action, rf_uc_object2valid_action, uo_object2valid_action, uv_object2valid_action


if opt.zero_shot_type == 'ua':
    limited_object2valid_action = ua_object2valid_action
if opt.zero_shot_type == 'nf_uc':
    limited_object2valid_action = nf_uc_object2valid_action
if opt.zero_shot_type == 'rf_uc':
    limited_object2valid_action = rf_uc_object2valid_action
if opt.zero_shot_type == 'uo':
    limited_object2valid_action = uo_object2valid_action
if opt.zero_shot_type == 'uv':
    limited_object2valid_action = uv_object2valid_action

# path to save the trained classifier best checkpoint
path = f'data_{opt.zero_shot_type}/unseen_Classifier.pth'


att = np.load(opt.action_embedding)
att/=LA.norm(att, ord=2)
att = torch.tensor(att)

unseen_classifier = ClsUnseenTrain(att).cuda()


seen_features = np.load(f"data_{opt.zero_shot_type}/hoi_feats.npy")
seen_labels = np.load(f"data_{opt.zero_shot_type}/hoi_actions.npy")
seen_obj = np.load(f"data_{opt.zero_shot_type}/hoi_objs.npy")


inds = np.random.permutation(np.arange(len(seen_labels)))
total_train_examples = int (0.8 * len(seen_labels))
train_inds = inds[:total_train_examples]
test_inds = inds[total_train_examples:]


len(test_inds)+len(train_inds), len(seen_labels)

train_feats = seen_features[train_inds]
train_labels = seen_labels[train_inds]
train_obj = seen_obj[train_inds]
test_feats = seen_features[test_inds]
test_labels = seen_labels[test_inds]
test_obj = seen_obj[test_inds]



# bg_inds = np.where(seen_labels==0)
# fg_inds = np.where(seen_labels>0)


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



seen_labels.shape


dataset_train = Featuresdataset(train_feats, train_labels, train_obj)
dataloader_train = DataLoader(dataset_train, batch_size=512, shuffle=True) 
dataset_test = Featuresdataset(test_feats, test_labels, test_obj)
dataloader_test = DataLoader(dataset_test, batch_size=2048, shuffle=True) 


from torch.optim.lr_scheduler import StepLR

criterion = nn.BCELoss()

optimizer = optim.SGD(unseen_classifier.parameters(), lr=1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)



min_val_loss = float("inf")


def val():
    running_loss = 0.0
    global min_val_loss
    unseen_classifier.eval()
    for i, (inputs, labels, objs) in enumerate(dataloader_test, 0):
        inputs = inputs.to(torch.float32).cuda()
        labels = labels.to(torch.float32).cuda()

        logits = unseen_classifier(inputs)
        prior = [limited_object2valid_action(i) for i in objs] # ****
        prior = np.array(prior)
        prior = torch.tensor(prior, dtype=torch.float32).cuda()

        outputs = torch.mul(torch.sigmoid(logits), prior)

        loss = criterion(outputs, labels)

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'Test Loss {epoch + 1}, [{i + 1} / {len(dataloader_test)}], {(running_loss / i) :0.4f}')
    
    print(f'running loss {running_loss / (i+1)}')
    if (running_loss / (i+1)) < min_val_loss:
        min_val_loss = running_loss / (i+1)
        state_dict = unseen_classifier.state_dict()   
        torch.save(state_dict, path)
        print(f'saved {min_val_loss :0.4f}')



# In[ ]:


for epoch in range(400):
    unseen_classifier.train()
    running_loss = 0.0
    for i, (inputs, labels, objs) in enumerate(dataloader_train, 0):
        inputs = inputs.to(torch.float32).cuda()
        labels = labels.to(torch.float32).cuda()
        
        optimizer.zero_grad()

        logits = unseen_classifier(inputs)
        prior = [limited_object2valid_action(i) for i in objs]
        prior = np.array(prior)
        prior = torch.tensor(prior, dtype=torch.float32).cuda()

        outputs = torch.mul(torch.sigmoid(logits), prior)
       # outputs = torch.mul(outputs, mask.unsqueeze(0).expand(len(objs), -1).cuda())

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999: 
            print(f'Train Loss {epoch + 1}, [{i + 1} / {len(dataloader_train)}], {(running_loss / i) :0.4f}')
    val()
    scheduler.step()

    
print('Finished Training')


# CUDA_VISIBLE_DEVICES=6 python train_unseen_classifier.py --zero_shot_type nf_uc
# 不同设置要更改rf_uc_object2valid_action，和arguments
