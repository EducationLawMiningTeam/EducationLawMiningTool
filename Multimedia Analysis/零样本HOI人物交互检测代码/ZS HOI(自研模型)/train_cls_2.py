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
from index2some import unsenn_ua_object2action,  object2valid_action, unsenn_nf_uc_object2action, unsenn_rf_uc_object2action, nf_uc_object2valid_action

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
            self.unseen_object2valid_action = unsenn_ua_object2action
        if opt.zero_shot_type == 'nf_uc':
            self.unseen_object2valid_action = unsenn_nf_uc_object2action
        if opt.zero_shot_type == 'rf_uc':
            self.unseen_object2valid_action = unsenn_rf_uc_object2action
        

        
        self.classifier = ClsModelTrain(117)
        if opt.zero_shot_type == 'nf_uc':
            box_pair_predictor = torch.load('data_nf_uc/box_pair_predictor.pth')
        if opt.zero_shot_type == 'rf_uc':
            box_pair_predictor = torch.load('data_rf_uc/box_pair_predictor.pth')
        self.classifier.fc1.load_state_dict(box_pair_predictor.state_dict())
        self.classifier.cuda()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.opt.lr_cls, betas=(0.5, 0.999))


    def init_model(self):
        pass

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

                prior = [self.unseen_object2valid_action(i) for i in in_obj] 
                prior = np.array(prior)
                prior = torch.tensor(prior, dtype=torch.float32).cuda()
                scores = torch.mul(torch.sigmoid(logits), prior) 
                loss = self.criterion(scores, in_action)

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

            # self.valacc, self.all_acc, c_mat_test = val(self.test_dataloader, self.classifier, self.criterion, self.opt, epoch, verbose="Test")
            # self.val_accuracies.append(self.all_acc)

            # if self.best_acc <= self.valacc:
            #     torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best.pth")
            #     print(f"saved best model best accuracy : {self.valacc:0.6f}")
            #     self.isBestIter = True
            #     np.save(f'{self.opt.outname}/confusion_matrix_Test.npy', c_mat_test)
            # self.best_acc = max(self.best_acc, self.valacc)
            # if self.isBestIter:
            #     self.best_epoch = self.gan_epoch
            #     torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best_latest.pth")
        
        # _,_, c_mat_train = compute_per_class_acc(np.concatenate(gt_all), np.concatenate(preds_all), self.opt, verbose='Train')
        # np.save(f'{self.opt.outname}/confusion_matrix_Train.npy', c_mat_train)
            with torch.no_grad():
                # box_pair_predictor = torch.load('data_ua/box_pair_predictor.pth')


                # torch.save(self.box_pair_predictor, f"results_{self.opt.zero_shot_type}_{self.opt.use_type}/classifier_latest_199_{epoch+1}.pth")
                torch.save(self.classifier.fc1, f"classifier_latest_199_{epoch+1}.pth")

        print(f"[{self.best_epoch:04}] best model accuracy {self.best_acc}")
