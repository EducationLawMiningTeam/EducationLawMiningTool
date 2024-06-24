# 把action object hoi一并传入

from __future__ import print_function
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import math
from util import *
import model
from cls_models import ClsModelTrain, ClsUnseenTrain
from index2some import hoi2action, hoi2obj, object2valid_action, unsenn_ua_object2action, unsenn_nf_uc_object2action, unsenn_rf_uc_object2action, unsenn_uv_object2action
from numpy import linalg as LA

def index_to_one_hot(index, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[index] = 1
    return one_hot

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TrainGAN():
    def __init__(self, opt, attributes, unseenAtt, unseenLabels, seen_feats_mean, gen_type='FG'):
        
        '''
        CLSWGAN trainer
        Inputs:
            opt -- arguments
            unseenAtt -- embeddings vector of unseen classes
            unseenLabels -- labels of unseen classes
            attributes -- embeddings vector of all classes
        '''
        self.opt = opt

        self.gen_type = gen_type

        # 原来的代码，我反正都没用
        # print(f"Wu_Labels {self.Wu_Labels}")
        # self.Wu = unseenAtt

        att = np.load(opt.action_embedding)
        att/=LA.norm(att, ord=2)
        att = torch.tensor(att)

        self.unseen_classifier = ClsUnseenTrain(att)
        self.unseen_classifier.cuda()
        self.unseen_classifier = loadUnseenWeights(f'data_{opt.zero_shot_type}/unseen_Classifier.pth', self.unseen_classifier)

        self.classifier = ClsModelTrain(num_classes=opt.nclass_all)   # 之前一直用ClsModel,这次改为ClsModelTrain
        self.classifier.cuda()
        self.classifier = loadhoipredictor(f'data_{opt.zero_shot_type}/box_pair_predictor.pth', self.classifier)

        self.suppressor = torch.load(f'data_{opt.zero_shot_type}/box_pair_suppressor.pth').cuda()

        
        self.unseenLabels = unseenLabels
        self.unseenLabels_obj = [hoi2obj(label) for label in self.unseenLabels]
        self.unseenAtt = unseenAtt

        self.criterion_obj = nn.CrossEntropyLoss()

        # # 创建模型实例
        # self.object_classification = Classifier(input_size=2048, num_classes=80).to("cuda")

        # # 加载保存的模型参数
        # self.object_classification.load_state_dict(torch.load(f'data_{opt.zero_shot_type}/classifier_model.pth'))


        for p in self.classifier.parameters():
            p.requires_grad = False
        
        for p in self.unseen_classifier.parameters():
            p.requires_grad = False

        for p in self.suppressor.parameters():
            p.requires_grad = False

        # for p in self.object_classification.parameters():
        #     p.requires_grad = False


        self.ntrain = opt.gan_epoch_budget
        self.attributes = attributes.data.numpy()

        print(f"# of training samples: {self.ntrain}")
        # initialize generator and discriminator
        self.netG = model.MLP_G(self.opt)
        self.netD = model.MLP_CRITIC(self.opt)


        if self.opt.cuda and torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()

        print('\n\n#############################################################\n')
        print(self.netG, '\n')
        print(self.netD)
        print('\n#############################################################\n\n')

        # classification loss, Equation (4) of the paper
        self.cls_criterion = nn.NLLLoss()
        self.my_criterion = nn.BCELoss()

        self.one = torch.tensor(1, dtype=torch.float)
        self.mone = self.one * -1

        if self.opt.cuda:
            
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.cls_criterion.cuda()


        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


        if opt.zero_shot_type == 'ua':
            self.unseen_object2valid_action = unsenn_ua_object2action
        if opt.zero_shot_type == 'nf_uc':
            self.unseen_object2valid_action = unsenn_nf_uc_object2action
        if opt.zero_shot_type == 'rf_uc':
            self.unseen_object2valid_action = unsenn_rf_uc_object2action
        if opt.zero_shot_type == 'uv':
            self.unseen_object2valid_action = unsenn_uv_object2action

    def __call__(self, epoch, features, labels, actions, objs):
        """
        Train GAN for one epoch
        Inputs:
            epoch: current epoch
            features: current epoch subset of features
            labels: ground truth labels
        """
        self.epoch = epoch
        self.features = features
        self.labels = labels
        self.actions = actions
        self.objs = objs
        self.ntrain = len(self.labels)
        self.trainEpoch()
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.netG)
        self.netG.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        self.netD.load_state_dict(torch.load(self.opt.netD)['state_dict'])
        print(f"loaded weights from epoch: {epoch} \n{self.opt.netD} \n{self.opt.netG} \n")
        return epoch
    def save_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netD.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc_{state}.pth')
        torch.save({'state_dict': self.netG.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen_{state}.pth')

    def generate_syn_feature(self, labels, attribute, objs, num=10, no_grad=True):
        # 这里要进行大幅度的修改，只返回action和feaure应该够了，不返回hoi_label (还要返回obj)
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of HOI
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each objectlabe在训练gan时generate_syn_feature 中no_grad=True
        returns:
            1) synthesised features 
            2) labels of synthesised  features 
        """

        nclass = labels.shape[0]
        syn_feature = torch.FloatTensor(nclass * num , self.opt.resSize)
        syn_label = torch.LongTensor(nclass*num)

        syn_obj = torch.LongTensor(nclass*num)
        syn_action = torch.FloatTensor(nclass * num , 117)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)
        
        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    obj = objs[i]
                    action_index = hoi2action(label)
                    action = torch.zeros(117)
                    action[action_index] = 1

                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise.normal_(0, 1)
                    output = self.netG(Variable(syn_noise), Variable(syn_att))
                
                    syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                    syn_label.narrow(0, i*num, num).fill_(label)
                    syn_obj.narrow(0, i*num, num).fill_(obj)

                    syn_action.narrow(0, i*num, num).copy_(action.data.cpu())

        else:
            for i in range(nclass):
                label = labels[i]
                obj = objs[i]
                action_index = hoi2action(label)
                action = torch.zeros(117)
                action[action_index] = 1

                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                output = self.netG(Variable(syn_noise), Variable(syn_att))
            
                syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(label)
                syn_obj.narrow(0, i*num, num).fill_(obj)

                syn_action.narrow(0, i*num, num).copy_(action.data.cpu())

        return syn_feature, syn_action, syn_obj
    

    def generate_syn_feature_2(self, labels, attribute, objs, num=10, no_grad=True):
        # 这里和generate_syn_feature几乎一摸一样，只不过返回了hoi_label
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of HOI
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each objectlabe在训练gan时generate_syn_feature 中no_grad=True
        returns:
            1) synthesised features 
            2) labels of synthesised  features 
        """

        nclass = labels.shape[0]
        syn_feature = torch.FloatTensor(nclass * num , self.opt.resSize)
        syn_label = torch.LongTensor(nclass*num)

        syn_obj = torch.LongTensor(nclass*num)
        syn_action = torch.FloatTensor(nclass * num , 117)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)
        
        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    obj = objs[i]
                    action_index = hoi2action(label)
                    action = torch.zeros(117)
                    action[action_index] = 1

                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise.normal_(0, 1)
                    output = self.netG(Variable(syn_noise), Variable(syn_att))
                
                    syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                    syn_label.narrow(0, i*num, num).fill_(label)
                    syn_obj.narrow(0, i*num, num).fill_(obj)

                    syn_action.narrow(0, i*num, num).copy_(action.data.cpu())

        else:
            for i in range(nclass):
                label = labels[i]
                obj = objs[i]
                action_index = hoi2action(label)
                action = torch.zeros(117)
                action[action_index] = 1

                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                output = self.netG(Variable(syn_noise), Variable(syn_att))
            
                syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(label)
                syn_obj.narrow(0, i*num, num).fill_(obj)

                syn_action.narrow(0, i*num, num).copy_(action.data.cpu())

        return syn_feature, syn_action, syn_obj



    def sample(self):
        """
        randomaly samples one batch of data
        returns (1)real features, (2)labels (3) attributes embeddings
        """
        idx = torch.randperm(self.ntrain)[0:self.opt.batch_size]
        # batch_feature = torch.from_numpy(self.features[idx])
        # batch_label = torch.from_numpy(self.labels[idx])
        # batch_action = torch.from_numpy(self.actions[idx])
        # batch_obj = torch.from_numpy(self.objs[idx])
        
        batch_feature = self.features[idx]
        batch_label = self.labels[idx]
        batch_action = self.actions[idx]
        batch_obj = self.objs[idx]


        batch_label2 = [lable.cpu().numpy().tolist().index(1) for lable in batch_label]  # 这里更改是因为只取一个hoi
 #       batch_att = [self.attributes[lable] for lable in batch_lable2]
        batch_att = torch.from_numpy(self.attributes[batch_label2])
        if 'BG' == self.gen_type:
            batch_label*=0
        return batch_feature, batch_label, batch_action, batch_obj, batch_att

    def calc_gradient_penalty(self, real_data, fake_data, input_att):
        alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates, Variable(input_att))

        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty

    def get_z_random(self):
        """
        returns normal initialized noise tensor 
        """
        z = torch.cuda.FloatTensor(self.opt.batch_size, self.opt.nz)
        z.normal_(0, 1)
        return z

    def trainEpoch(self):
        for i in range(0, self.ntrain, self.opt.batch_size):

            # import pdb; pdb.set_trace()
            input_res, input_label, input_action, input_obj, input_att = self.sample()

            if self.opt.batch_size != input_res.shape[0]:
                continue 

            input_res, input_label, input_action, input_obj, input_att =  input_res.type(torch.FloatTensor).cuda(), input_label.type(torch.LongTensor).cuda(), input_action.type(torch.FloatTensor).cuda(), input_obj.type(torch.LongTensor).cuda(), input_att.type(torch.FloatTensor).cuda()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            input_actionv = Variable(input_action)
            input_objv = Variable(input_obj)

            ############################
            # (1) Update D network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
            
            for p in self.netG.parameters(): # reset requires_grad
                p.requires_grad = False 
            

            for iter_d in range(self.opt.critic_iter):
                self.netD.zero_grad()
                # train with realG
                # sample a mini-batch
                # sparse_real = self.opt.resSize - input_res[1].gt(0).sum()

                criticD_real = self.netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(self.mone)
          #      print(input_resv)


                noise = self.get_z_random()
                noisev = Variable(noise)
                fake = self.netG(noisev, input_attv)

                criticD_fake = self.netD(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(self.one)
       #         print(fake)

                # print(torch.max(fake), torch.max(input_resv))
                # print(torch.min(fake), torch.min(input_resv))
                # print(torch.var(fake), torch.var(input_resv))
                

                # gradient penalty
                gradient_penalty = self.calc_gradient_penalty(input_res, fake.data, input_att)
                gradient_penalty.backward()

            #    gradient_penalty=0
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty 
          #      print(criticD_fake, criticD_real, criticD_fake - criticD_real)
            #    D_cost.backward()

                self.optimizerD.step()


            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation

            for p in self.netG.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            self.netG.zero_grad()
            input_attv = Variable(input_att)
            noise = self.get_z_random()
            noisev = Variable(noise)
            fake = self.netG(noisev, input_attv)
            criticG_fake = self.netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = criticG_fake 

            # ---------------------
            # mode seeking loss https://github.com/HelenMao/MSGAN/blob/e386c252f059703fcf5d439258949ae03dc46519/DCGAN-Mode-Seeking/model.py#L66
            noise2 = self.get_z_random()

            noise2v = Variable(noise2)
            fake2 = self.netG(noise2v, input_attv)

            lz = torch.mean(torch.abs(fake2 - fake)) / torch.mean(
                torch.abs(noise2v - noisev))
            eps = 1 * 1e-5
            loss_lz = 1 / (lz + eps)
            loss_lz*=self.opt.lz_ratio
            # ---------------------

            # classification loss
            # seen loss
            logits1 = self.classifier(feats=fake, classifier_only=True)
            suppressor1 = self.suppressor(fake)
            suppressor1 = torch.sigmoid(suppressor1)
            suppressor1 = suppressor1.expand(self.opt.batch_size, 117)
            prior1 = [object2valid_action(i) for i in input_objv]
            prior1 = np.array(prior1)
            prior1 = torch.tensor(prior1, dtype=torch.float32).cuda()
            scores1 = torch.mul(torch.sigmoid(logits1) * suppressor1, prior1)
            c_errG = self.my_criterion(scores1, Variable(input_actionv))
            
            c_errG = self.opt.cls_weight*c_errG


            # object classification loss
            # pred_obj = self.object_classification(fake)  # 在seen上加效果不大
            # obj_loss = self.criterion_obj(pred_obj, input_objv)


            # with torch.no_grad():  # 只是想检查一下真特征对应的loss
              #  # action_loss
                # real_logits = self.classifier(feats=input_resv, classifier_only=True)
                # real_suppressor = self.suppressor(input_resv)
                # real_suppressor = torch.sigmoid(real_suppressor)
                # real_suppressor = real_suppressor.expand(self.opt.batch_size, 117)
                # real_prior = [object2valid_action(i) for i in input_objv]
                # real_prior = np.array(real_prior)
                # real_prior = torch.tensor(real_prior, dtype=torch.float32).cuda()
                # real_scores = torch.mul(torch.sigmoid(real_logits) * real_suppressor, real_prior)
                # loss_real = self.my_criterion(real_scores, Variable(input_actionv))

            #    # obj_loss
            # obj = self.object_classification(input_resv)
            # real_obj_loss = self.criterion_obj(obj, input_objv)
            # print(real_obj_loss)


            # 显示一下中间结果
            # pp = [find_indices(p, 1) for p in prior1]   
            # for h, p in enumerate(pp):
            #     print('fake:', scores1[h, p].cpu().detach().numpy())
            #     print('real:', real_scores[h, p].cpu().detach().numpy())
            #     print('gt:', input_actionv[h, p].cpu().detach().numpy())

            #     print('/n/n/n')

            #     print('fake:', input_resv[h])
            #     print('real:', fake[h])

            # --------------------------------------------

            # unseen action loss
            fake_unseen_f, fake_unseen_act, fake_unseen_obj = self.generate_syn_feature(self.unseenLabels, self.unseenAtt, self.unseenLabels_obj, num=self.opt.batch_size//32, no_grad=False)

            logits2 = self.unseen_classifier(feats=fake_unseen_f.cuda(), classifier_only=True)
            prior2 = [self.unseen_object2valid_action(i) for i in fake_unseen_obj]
            prior2 = np.array(prior2)
            prior2 = torch.tensor(prior2, dtype=torch.float32).cuda()
            scores2 = torch.mul(torch.sigmoid(logits2), prior2)

            unseenc_errG = self.opt.cls_weight_unseen * self.my_criterion(scores2, Variable(fake_unseen_act.cuda()))



            # unseen obj loss 
            # fake_unseen_f, fake_unseen_act, fake_unseen_obj = self.generate_syn_feature(self.unseenLabels, self.unseenAtt, self.unseenLabels_obj, num=self.opt.batch_size//32, no_grad=False) 
            # pred_obj = self.object_classification(fake_unseen_f.cuda()) 
            # unseen_obj_loss = 2 * self.criterion_obj(pred_obj, fake_unseen_obj.cuda())

            

            # ---------------------------------------------
            # Total loss 


  #          errG = -G_cost + loss_lz + c_errG + unseen_obj_loss + obj_loss
            
            errG =  -G_cost + c_errG + loss_lz + unseenc_errG
     #       c_errG.backward(retain_graph=True)
            errG.backward()
            self.optimizerG.step()

            # print(f"{self.gen_type} [{self.epoch+1:02}/{self.opt.nepoch:02}] [{i:06}/{int(self.ntrain)}] \
            # Loss: {errG.item() :0.4f}  D loss: {D_cost.data.item():.4f} G loss: {G_cost.data.item():.4f}, W dist: {Wasserstein_D.data.item():.4f} \
            # seen loss: {c_errG:.4f}  unseen loss: {unseenc_errG:.4f}  loss div: {loss_lz.item():0.4f}")

            print(f"{self.gen_type} [{self.epoch+1:02}/{self.opt.nepoch:02}] [{i:06}/{int(self.ntrain)}] \
            Loss: {errG.item() :0.4f}  D loss: {D_cost.data.item():.4f} G loss: {G_cost.data.item():.4f}, W dist: {Wasserstein_D.data.item():.4f} \
            seen loss: {c_errG:.4f}  unseen loss: {unseenc_errG:.4f}  loss div: {loss_lz.item():0.4f}")
            
        self.netG.eval()


def find_indices(lst, value):
    indices = []
    for i, x in enumerate(lst):
        if x == value:
            indices.append(i)
    return indices
