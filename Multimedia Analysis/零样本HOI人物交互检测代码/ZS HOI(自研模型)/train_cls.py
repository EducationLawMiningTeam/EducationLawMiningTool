from __future__ import print_function
import torch
import torch.optim as optim
from util import *
import torch.nn as nn
# from mmdetection.tools.faster_rcnn_utils import *
from torch.utils.data import DataLoader
import numpy as np
from hoi_dataset import * 
from cls_models import ClsModelTrain
from mmdetection.splits import get_unseen_class_ids, get_unseen_class_labels 
from rare_and_no import get_rare
from index2some import unsenn_ua_object2action,  object2valid_action

class TrainHoiCls():
    def __init__(self, opt):

        # self.classes_to_train = get_rare()
        self.opt = opt
        # self.classes = get_rare()
        self.best_acc = -100000
        self.isBestIter = False
    #    self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss()

        self.dataset = None 
        self.val_accuracies = []
        self.init_model()
        self.best_epoch = 0
        if opt.zero_shot_type == 'ua':
            self.ua_uv = [8, 9,  25,  31,  37,  38,  40,  46,  47,  65,  67,  70,  72,  78,  80,  85,  86,  89, 93,  97,  98, 103]
            self.box_pair_predictor = torch.load('data_ua/box_pair_predictor.pth')
        if opt.zero_shot_type == 'uv':
            self.ua_uv = [41, 100, 99, 91, 34, 42, 97, 84, 26, 106, 38, 56, 92, 79, 19, 76, 80, 2, 114, 62]
            self.box_pair_predictor = torch.load('data_uv/box_pair_predictor.pth')


    def init_model(self):
        self.classifier = ClsModelTrain(117)

        # box_pair_predictor = torch.load('data_ua/box_pair_predictor.pth')
        # self.classifier.fc1.load_state_dict(box_pair_predictor.state_dict())

        self.classifier.cuda()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.opt.lr_cls, betas=(0.5, 0.999))

    def initDataSet(self, features, actions, objs):
        self.dataset = FeaturesCls(self.opt, features=features, actions=actions, objs=objs) #, split='train', classes_to_train=self.classes_to_train)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.batch_size, num_workers=4, shuffle=True, pin_memory=True)
        
    def updateDataSet(self, features, actions):
        self.dataloader.dataset.replace(features, actions)

    def __call__(self, features, actions, objs, gan_epoch=0):
        self.isBestIter = False
        self.gan_epoch = gan_epoch

        if self.dataset is None:
            self.initDataSet(features, actions, objs)
        
        self.init_model()
        self.trainEpochs()

    def trainEpochs(self):
        for epoch in range(self.opt.nepoch_cls):
            self.classifier.train()
            loss_epoch = 0
            scores_all = []
            gt_all = []
            for ite, (in_feat, in_action, in_obj)  in enumerate(self.dataloader):
                in_feat = in_feat.type(torch.float).cuda()
                in_action = in_action.cuda()
                logits = self.classifier(feats=in_feat, classifier_only=True)

                prior = [unsenn_ua_object2action(i) for i in in_obj]
                prior = np.array(prior)
                prior = torch.tensor(prior, dtype=torch.float32).cuda()
                scores = torch.mul(torch.sigmoid(logits), prior)   # 应该要用
                loss = self.criterion(torch.sigmoid(logits), in_action)

                loss_epoch+=loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # scores_all.append(scores_all.data.cpu().numpy())
                # gt_all.append(in_action.data.cpu().numpy())
                
                if ite % 30 == 29:
                    print(f'Cls Train Epoch [{epoch+1:02}/{self.opt.nepoch_cls}] Iter [{ite:05}/{len(self.dataloader)}]{ite/len(self.dataloader) * 100:02.3f}% Loss: {loss_epoch/ite :0.4f} lr: {get_lr(self.optimizer):0.6f}')
            # validate on test set
            adjust_learning_rate(self.optimizer, epoch, self.opt)

            with torch.no_grad():
                for unsenn_verb in self.ua_uv:
                    self.box_pair_predictor.weight[unsenn_verb] = self.classifier.fc1.weight.clone()[unsenn_verb]
                    self.box_pair_predictor.bias[unsenn_verb] = self.classifier.fc1.bias.clone()[unsenn_verb]

                torch.save(self.box_pair_predictor, f"results_{self.opt.zero_shot_type}_{self.opt.use_type}/classifier_latest_269_{epoch+1}.pth")

        print(f"[{self.best_epoch:04}] best model accuracy {self.best_acc}")
