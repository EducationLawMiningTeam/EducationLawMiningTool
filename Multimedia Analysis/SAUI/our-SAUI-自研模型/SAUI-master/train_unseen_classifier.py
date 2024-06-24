from cls_models import ClsUnseenTrain
from generate import load_seen_att
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from util import loadUnseenWeights
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

eval_only = 0
opt = dotdict({
    'dataset':'coco',
    'classes_split': '65_15',
    'class_embedding': 'MSCOCO/cliptextfeature.npy',
    'dataroot':'./mmdetection/feature/',
    'trainsplit': 'train_0.6_0.3',
    
})
save_root = 'MSCOCO'
save_path = f"{save_root}/unseen_Classifier.pth"


class Featuresdataset(Dataset):
     
    def __init__(self, features, labels, classid_tolabels):
        self.classid_tolabels = classid_tolabels
        self.features = features
        self.labels = labels
        

    def __getitem__(self, idx):
        batch_feature = self.features[idx]
        batch_label = self.labels[idx]
#         import pdb; pdb.set_trace()
        
        if self.classid_tolabels is not None:
            batch_label = self.classid_tolabels[batch_label]
        return batch_feature, batch_label

    def __len__(self):
        return len(self.labels)
    
def val():
    running_loss = 0.0
    global min_val_loss
    unseen_classifier.eval()
    for i, (inputs_tem, labels) in enumerate(dataloader_test, 0):
        inputs = torch.cat((inputs_tem[:,0], inputs_tem[:,1], inputs_tem[:,2], inputs_tem[:,3]), dim=1)
        inputs = inputs.cuda()
        labels = labels.cuda()
        

        outputs = unseen_classifier(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        if i % 70 == 69:
            print(f'Test Loss {epoch + 1}, [{i + 1} / {len(dataloader_test)}], {(running_loss / i) :0.4f}')
    if (running_loss / i) < min_val_loss:
        min_val_loss = running_loss / i
        state_dict = unseen_classifier.state_dict()   
        torch.save(state_dict, save_path)
        print(f'saved {min_val_loss :0.4f}')




seen_att, att_labels = load_seen_att(opt)
classid_tolabels = {l:i for i, l in enumerate(att_labels.data.numpy())}

unseen_classifier = ClsUnseenTrain(seen_att).cuda()
print("UnseenCLSModel Ready!")

# path to save the trained classifier best checkpoint

if save_root is not None:
    try:
        os.makedirs(save_root)
    except OSError:
        pass

seen_features = np.load(f"{opt.dataroot}/{opt.trainsplit}_feats.npy")
seen_labels = np.load(f"{opt.dataroot}/{opt.trainsplit}_labels.npy")

inds = np.random.permutation(np.arange(len(seen_labels)))
total_train_examples = int (0.8 * len(seen_labels))
train_inds = inds[:total_train_examples]
test_inds = inds[total_train_examples:]

train_feats = seen_features[train_inds]
train_labels = seen_labels[train_inds]
test_feats = seen_features[test_inds]
test_labels = seen_labels[test_inds]

dataset_train = Featuresdataset(train_feats, train_labels, classid_tolabels)
dataloader_train = DataLoader(dataset_train, batch_size=512, shuffle=True) 
dataset_test = Featuresdataset(test_feats, test_labels, classid_tolabels)
dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=True) 
print("DataLoader Ready!")
if eval_only :
    val()
else :
    from torch.optim.lr_scheduler import StepLR

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(unseen_classifier.parameters(), lr=1, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    min_val_loss = float("inf")
    print("Training Procedure Started!")
    for epoch in range(100):
        unseen_classifier.train()
        running_loss = 0.0
        for i, (inputs_tem, labels) in enumerate(dataloader_train, 0):
            inputs = torch.cat((inputs_tem[:,0], inputs_tem[:,1], inputs_tem[:,2], inputs_tem[:,3]), dim=1)
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()

            outputs = unseen_classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99: 
                print(f'Train Loss {epoch + 1}, [{i + 1} / {len(dataloader_train)}], {(running_loss / i) :0.4f}')
        val()
        scheduler.step()
        
    print('Finished Training')

