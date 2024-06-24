from __future__ import print_function
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import math
from util import *
import model
from cls_models import ClsModel, ClsUnseen
import random
import losses
import torch.nn.functional as F

class TrainGAN():
    def __init__(self, opt, attributes, unseenAtt, unseenLabels, gen_type='FG'):
        
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
        self.Wu_Labels = np.array([i for i, l in enumerate(unseenLabels)])
        print(f"Wu_Labels {self.Wu_Labels}")
        self.Wu = unseenAtt

        self.unseen_classifier = ClsUnseen(unseenAtt)
        self.unseen_classifier.cuda()

        self.unseen_classifier = loadUnseenWeights(opt.pretrain_classifier_unseen, self.unseen_classifier)
        self.classifier = ClsModel(num_classes=opt.nclass_all)
        self.classifier.cuda()
        self.classifier = loadFasterRcnnCLSHead(opt.pretrain_classifier, self.classifier)
        
        for p in self.classifier.parameters():
            p.requires_grad = False
        
        for p in self.unseen_classifier.parameters():
            p.requires_grad = False

        self.ntrain = opt.gan_epoch_budget
        self.attributes = attributes.data.numpy()

        print(f"# of training samples: {self.ntrain}")
        # initialize generator and discriminator
        ###########
        self.netG3 = model.MLP_G(self.opt)
        self.netG2 = model.MLP_CG(self.opt)
        self.netG1 = model.MLP_CG(self.opt)
        self.netG0 = model.MLP_CG(self.opt)
        self.netD3 = model.MLP_CRITIC(self.opt)
        self.netD2 = model.MLP_CRITIC(self.opt)
        self.netD1 = model.MLP_CRITIC(self.opt)
        self.netD0 = model.MLP_CRITIC(self.opt)
        ###########
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.opt.featnorm = True
        self.inter_contras_criterion = losses.SupConLoss_clear(self.opt.alpha_main)
        self.multiscale_contras_criterion = losses.SupConLoss_clear(self.opt.tau)

        if self.opt.cuda and torch.cuda.is_available():
            self.netG3 = self.netG3.cuda()
            self.netG2 = self.netG2.cuda()
            self.netG1 = self.netG1.cuda()
            self.netG0 = self.netG0.cuda()
            self.netD3 = self.netD3.cuda()
            self.netD2 = self.netD2.cuda()
            self.netD1 = self.netD1.cuda()
            self.netD0 = self.netD0.cuda()

        print('\n\n#############################################################\n')
        print('Multiscale Feature Synthesizer initialized.')
        print('\n#############################################################\n\n')

        # classification loss, Equation (4) of the paper
        self.cls_criterion = nn.NLLLoss()

        self.one = torch.tensor(1, dtype=torch.float)
        self.mone = self.one * -1

        if self.opt.cuda:
            
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.cls_criterion.cuda()
            self.cross_entropy_loss.cuda()
            self.inter_contras_criterion.cuda()


        self.optimizerD3 = optim.Adam(self.netD3.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerD2 = optim.Adam(self.netD2.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerD1 = optim.Adam(self.netD1.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerD0 = optim.Adam(self.netD0.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG3 = optim.Adam(self.netG3.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG2 = optim.Adam(self.netG2.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG1 = optim.Adam(self.netG1.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG0 = optim.Adam(self.netG0.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def __call__(self, epoch, features, labels):
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
        self.ntrain = len(self.labels)
        self.trainEpoch()
    
    def load_checkpoint(self, load_path, resume_train):
        """
        
        """
        load = 'latest'
        if resume_train == 1:
            load = 'best'
        elif resume_train == 2:
            load = 'latest'
        checkpoint = torch.load(load_path+f"gen3_{load}.pth")
        self.netG3.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(load_path+f"gen2_{load}.pth")
        self.netG2.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(load_path+f"gen1_{load}.pth")
        self.netG1.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(load_path+f"gen0_{load}.pth")
        self.netG0.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        self.netD3.load_state_dict(torch.load(load_path+f"disc3_{load}.pth")['state_dict'])
        self.netD2.load_state_dict(torch.load(load_path+f"disc2_{load}.pth")['state_dict'])
        self.netD1.load_state_dict(torch.load(load_path+f"disc1_{load}.pth")['state_dict'])
        self.netD0.load_state_dict(torch.load(load_path+f"disc0_{load}.pth")['state_dict'])
        print(f"loaded weights from epoch: {epoch} \n")
        return epoch
    
    def save_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netD3.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc3_{state}.pth')
        torch.save({'state_dict': self.netD2.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc2_{state}.pth')
        torch.save({'state_dict': self.netD1.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc1_{state}.pth')
        torch.save({'state_dict': self.netD0.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc0_{state}.pth')
        torch.save({'state_dict': self.netG3.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen3_{state}.pth')
        torch.save({'state_dict': self.netG2.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen2_{state}.pth')
        torch.save({'state_dict': self.netG1.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen1_{state}.pth')
        torch.save({'state_dict': self.netG0.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen0_{state}.pth')

    def generate_syn_feature(self, labels, attribute, num=100, no_grad=True):
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of objects 
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each object
        returns:
            1) synthesised features [nclass * num , 4, self.opt.resSize]
            2) labels of synthesised  features 
        """

        nclass = labels.shape[0]
        syn_feature = torch.FloatTensor(nclass * num , 4, self.opt.resSize)
        syn_label = torch.LongTensor(nclass*num)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise3 = torch.FloatTensor(num, self.opt.nz)
        syn_noise2 = torch.FloatTensor(num, self.opt.nz)
        syn_noise1 = torch.FloatTensor(num, self.opt.nz)
        syn_noise0 = torch.FloatTensor(num, self.opt.nz)
        
        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise3 = syn_noise3.cuda()
            syn_noise2 = syn_noise2.cuda()
            syn_noise1 = syn_noise1.cuda()
            syn_noise0 = syn_noise0.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise3.normal_(0, 1)
                    syn_noise2.normal_(0, 1)
                    syn_noise1.normal_(0, 1)
                    syn_noise0.normal_(0, 1)
                    output3 = self.netG3(Variable(syn_noise3), Variable(syn_att))
                    output2 = self.netG2(Variable(syn_noise2), Variable(syn_att), output3),
                    output2 = output2[0]
                    output1 = self.netG1(Variable(syn_noise1), Variable(syn_att), output2),
                    output1 = output1[0]
                    output0 = self.netG0(Variable(syn_noise0), Variable(syn_att), output1),
                    output0 = output0[0]

                    syn_feature[:,3].narrow(0, i*num, num).copy_(output3.data.cpu())
                    syn_feature[:,2].narrow(0, i*num, num).copy_(output2.data.cpu())
                    syn_feature[:,1].narrow(0, i*num, num).copy_(output1.data.cpu())
                    syn_feature[:,0].narrow(0, i*num, num).copy_(output0.data.cpu())
                    syn_label.narrow(0, i*num, num).fill_(label)
        else:
            for i in range(nclass):
                label = labels[i]
                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise3.normal_(0, 1)
                syn_noise2.normal_(0, 1)
                syn_noise1.normal_(0, 1)
                syn_noise0.normal_(0, 1)
                output3 = self.netG3(Variable(syn_noise3), Variable(syn_att))
                output2 = self.netG2(Variable(syn_noise2), Variable(syn_att), output3),
                output2 = output2[0]
                output1 = self.netG1(Variable(syn_noise1), Variable(syn_att), output2),
                output1 = output1[0]
                output0 = self.netG0(Variable(syn_noise0), Variable(syn_att), output1),
                output0 = output0[0]

                syn_feature[:,3].narrow(0, i*num, num).copy_(output3.data.cpu())
                syn_feature[:,2].narrow(0, i*num, num).copy_(output2.data.cpu())
                syn_feature[:,1].narrow(0, i*num, num).copy_(output1.data.cpu())
                syn_feature[:,0].narrow(0, i*num, num).copy_(output0.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(label)

        return syn_feature, syn_label

    def sample(self):
        """
        randomaly samples one batch of data
        returns (1)real features, (2)labels (3) attributes embeddings
        """
        idx = torch.randperm(self.ntrain)[0:self.opt.batch_size]
        batch_feature = torch.from_numpy(self.features[idx])
        batch_label = torch.from_numpy(self.labels[idx])
        batch_att = torch.from_numpy(self.attributes[batch_label])
        if 'BG' == self.gen_type:
            batch_label*=0
        return batch_feature, batch_label, batch_att

    def calc_gradient_penalty(self, real_data, fake_data, input_att, scale_id, contra=False):
        if scale_id == 3:
            self.netD = self.netD3
        elif scale_id == 2:
            self.netD = self.netD2
        elif scale_id == 1:
            self.netD = self.netD1
        else:
            self.netD = self.netD0

        if contra:
            alpha = torch.rand(real_data.size(0), 1)
        else:
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
    
    def compute_contrastive_loss(self, feat_q, feat_k):

        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / self.opt.tau
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss
    
    def latent_augmented_sampling(self):

        query = self.get_z_random_v2(self.opt.batch_size, self.opt.nz*4, 'gauss')
        pos = torch.cuda.FloatTensor(query.shape).uniform_(-self.opt.radius, self.opt.radius).add_(query)
        negs = []
        for k in range(self.opt.num_negative):
            neg = self.get_z_random_v2(self.opt.batch_size, self.opt.nz*4, 'gauss')
            while (neg - query).abs().min() < self.opt.radius:
                neg = self.get_z_random_v2(self.opt.batch_size, self.opt.nz*4, 'gauss')
            negs.append(neg)
        negs = torch.cat(negs, 0)
        return query, pos, negs

    def get_z_random_v2(self, batchSize, nz, random_type='gauss'):
        z = torch.cuda.FloatTensor(batchSize, nz)
        z.copy_(torch.randn(batchSize, nz))
        return z
    
    def trainEpoch(self):

        for i in range(0, self.ntrain, self.opt.batch_size):
            # import pdb; pdb.set_trace()
            input_res, input_label, input_att = self.sample()

            if self.opt.batch_size != input_res.shape[0]:
                continue
            input_res, input_label, input_att = input_res.type(torch.FloatTensor).cuda(), input_label.type(torch.LongTensor).cuda(), input_att.type(torch.FloatTensor).cuda()
            ############################
            # (1) Update D network: optimize WGAN-GP objective
            ###########################
            for p in self.netD3.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
            for p in self.netD2.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
            for p in self.netD1.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
            for p in self.netD0.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update
            input_res3 = input_res[:,3]
            input_res2 = input_res[:,2]
            input_res1 = input_res[:,1]
            input_res0 = input_res[:,0]
            input_resv3 = Variable(input_res[:,3])
            input_resv2 = Variable(input_res[:,2])
            input_resv1 = Variable(input_res[:,1])
            input_resv0 = Variable(input_res[:,0])
            input_attv = Variable(input_att)

            for iter_d in range(self.opt.critic_iter):
                self.netD3.zero_grad()
                self.netD2.zero_grad()
                self.netD1.zero_grad()
                self.netD0.zero_grad()


                criticD3_real = self.netD3(input_resv3, input_attv)
                criticD2_real = self.netD2(input_resv2, input_attv)
                criticD1_real = self.netD1(input_resv1, input_attv)
                criticD0_real = self.netD0(input_resv0, input_attv)
                criticD3_real = criticD3_real.mean()
                criticD2_real = criticD2_real.mean()
                criticD1_real = criticD1_real.mean()
                criticD0_real = criticD0_real.mean()
                criticD3_real.backward(self.mone)
                criticD2_real.backward(self.mone)
                criticD1_real.backward(self.mone)
                criticD0_real.backward(self.mone)

                real_inter_contras_loss3 = self.inter_contras_criterion(F.normalize((input_resv3), dim=1), input_label).requires_grad_()
                real_inter_contras_loss2 = self.inter_contras_criterion(F.normalize((input_resv2), dim=1), input_label).requires_grad_()
                real_inter_contras_loss1 = self.inter_contras_criterion(F.normalize((input_resv1), dim=1), input_label).requires_grad_()
                real_inter_contras_loss0 = self.inter_contras_criterion(F.normalize((input_resv0), dim=1), input_label).requires_grad_()
                real_inter_contras_loss3.backward()
                real_inter_contras_loss2.backward()
                real_inter_contras_loss1.backward()
                real_inter_contras_loss0.backward()
                
                noise3 = self.get_z_random()
                noise2 = self.get_z_random()
                noise1 = self.get_z_random()
                noise0 = self.get_z_random()

                query, pos, negs = self.latent_augmented_sampling()
                query0 = query[:,:self.opt.nz]
                query1 = query[:,self.opt.nz:2*self.opt.nz]
                query2 = query[:,2*self.opt.nz:3*self.opt.nz]
                query3 = query[:,3*self.opt.nz:]
                pos0 = pos[:,:self.opt.nz]
                pos1 = pos[:,self.opt.nz:2*self.opt.nz]
                pos2 = pos[:,2*self.opt.nz:3*self.opt.nz]
                pos3 = pos[:,3*self.opt.nz:]
                negs0 = negs[:,:self.opt.nz]
                negs1 = negs[:,self.opt.nz:2*self.opt.nz]
                negs2 = negs[:,2*self.opt.nz:3*self.opt.nz]
                negs3 = negs[:,3*self.opt.nz:]
                noise3_ = torch.cat([query3, pos3, negs3],0)
                noise2_ = torch.cat([query2, pos2, negs2],0)
                noise1_ = torch.cat([query1, pos1, negs1],0)
                noise0_ = torch.cat([query0, pos0, negs0],0)

                z_conc3 = torch.cat([noise3, noise3_], 0)
                z_conc2 = torch.cat([noise2, noise2_], 0)
                z_conc1 = torch.cat([noise1, noise1_], 0)
                z_conc0 = torch.cat([noise0, noise0_], 0)
                attn_conc = torch.cat([input_attv] * (self.opt.num_negative + 3), 0)
                input_resv3_conc = torch.cat([input_resv3] * (self.opt.num_negative + 3), 0)
                input_resv2_conc = torch.cat([input_resv2] * (self.opt.num_negative + 3), 0)
                input_resv1_conc = torch.cat([input_resv1] * (self.opt.num_negative + 3), 0)

                fake3 = self.netG3(z_conc3, attn_conc)
                fake2 = self.netG2(z_conc2, attn_conc, input_resv3_conc)
                fake1 = self.netG1(z_conc1, attn_conc, input_resv2_conc)
                fake0 = self.netG0(z_conc0, attn_conc, input_resv1_conc)

                fake3_a = fake3[:input_resv3.size(0)]
                fake2_a = fake2[:input_resv2.size(0)]
                fake1_a = fake1[:input_resv1.size(0)]
                fake0_a = fake0[:input_resv0.size(0)]
                fake3_b = fake3[input_resv3.size(0):]
                fake2_b = fake2[input_resv2.size(0):]
                fake1_b = fake1[input_resv1.size(0):]
                fake0_b = fake0[input_resv0.size(0):]

                criticD3_fake = self.netD3(fake3_a.detach(), input_attv)
                criticD2_fake = self.netD2(fake2_a.detach(), input_attv)
                criticD1_fake = self.netD1(fake1_a.detach(), input_attv)
                criticD0_fake = self.netD0(fake0_a.detach(), input_attv)
                criticD3_fake = criticD3_fake.mean()
                criticD2_fake = criticD2_fake.mean()
                criticD1_fake = criticD1_fake.mean()
                criticD0_fake = criticD0_fake.mean()
                criticD3_fake.backward(self.one)
                criticD2_fake.backward(self.one)
                criticD1_fake.backward(self.one)
                criticD0_fake.backward(self.one)

                # gradient penalty
                gradient_penalty3 = self.calc_gradient_penalty(input_res[:,3], fake3_a.data, input_att, 3)
                gradient_penalty2 = self.calc_gradient_penalty(input_res[:,2], fake2_a.data, input_att, 2)
                gradient_penalty1 = self.calc_gradient_penalty(input_res[:,1], fake1_a.data, input_att, 1)
                gradient_penalty0 = self.calc_gradient_penalty(input_res[:,0], fake0_a.data, input_att, 0)
                gradient_penalty3.backward()
                gradient_penalty2.backward()
                gradient_penalty1.backward()
                gradient_penalty0.backward()

                D3_cost = criticD3_fake - criticD3_real + gradient_penalty3
                D2_cost = criticD2_fake - criticD2_real + gradient_penalty2
                D1_cost = criticD1_fake - criticD1_real + gradient_penalty1
                D0_cost = criticD0_fake - criticD0_real + gradient_penalty0
                D_cost_aver = (D3_cost.item() + D2_cost.item() + D1_cost.item() + D0_cost.item())/4
                
                criticD3_real2 = self.netD3(input_resv3, input_attv)
                criticD2_real2 = self.netD2(input_resv2, input_attv)
                criticD1_real2 = self.netD1(input_resv1, input_attv)
                criticD0_real2 = self.netD0(input_resv0, input_attv)
                criticD3_real2 = criticD3_real2.mean()
                criticD2_real2 = criticD2_real2.mean()
                criticD1_real2 = criticD1_real2.mean()
                criticD0_real2 = criticD0_real2.mean()
                criticD3_real2.backward(self.mone)
                criticD2_real2.backward(self.mone)
                criticD1_real2.backward(self.mone)
                criticD0_real2.backward(self.mone)

                self.netD3(fake3_b.detach(), input_attv.repeat(self.opt.num_negative + 2, 1)).mean().backward(self.one)
                self.netD2(fake2_b.detach(), input_attv.repeat(self.opt.num_negative + 2, 1)).mean().backward(self.one)
                self.netD1(fake1_b.detach(), input_attv.repeat(self.opt.num_negative + 2, 1)).mean().backward(self.one)
                self.netD0(fake0_b.detach(), input_attv.repeat(self.opt.num_negative + 2, 1)).mean().backward(self.one)

                self.calc_gradient_penalty(input_res[:,3].repeat(self.opt.num_negative + 2, 1),
                                                               fake3_b.data,
                                                               input_att.repeat(self.opt.num_negative + 2, 1),3,
                                                               contra=True).backward()
                self.calc_gradient_penalty(input_res[:,2].repeat(self.opt.num_negative + 2, 1),
                                                               fake2_b.data,
                                                               input_att.repeat(self.opt.num_negative + 2, 1),2,
                                                               contra=True).backward()
                self.calc_gradient_penalty(input_res[:,1].repeat(self.opt.num_negative + 2, 1),
                                                               fake1_b.data,
                                                               input_att.repeat(self.opt.num_negative + 2, 1),1,
                                                               contra=True).backward()
                self.calc_gradient_penalty(input_res[:,0].repeat(self.opt.num_negative + 2, 1),
                                                               fake0_b.data,
                                                               input_att.repeat(self.opt.num_negative + 2, 1),0,
                                                               contra=True).backward()
                
                self.optimizerD3.step()
                self.optimizerD2.step()
                self.optimizerD1.step()
                self.optimizerD0.step()

            ############################
            # (2) Update G network: optimize WGAN-GP objective
            ###########################
            for p in self.netD3.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation
            for p in self.netD2.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation
            for p in self.netD1.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation
            for p in self.netD0.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation

            self.netG3.zero_grad()
            self.netG2.zero_grad()
            self.netG1.zero_grad()
            self.netG0.zero_grad()

            noise3 = self.get_z_random()
            noise2 = self.get_z_random()
            noise1 = self.get_z_random()
            noise0 = self.get_z_random()

            query, pos, negs = self.latent_augmented_sampling()
            query0 = query[:,:self.opt.nz]
            query1 = query[:,self.opt.nz:2*self.opt.nz]
            query2 = query[:,2*self.opt.nz:3*self.opt.nz]
            query3 = query[:,3*self.opt.nz:]
            pos0 = pos[:,:self.opt.nz]
            pos1 = pos[:,self.opt.nz:2*self.opt.nz]
            pos2 = pos[:,2*self.opt.nz:3*self.opt.nz]
            pos3 = pos[:,3*self.opt.nz:]
            negs0 = negs[:,:self.opt.nz]
            negs1 = negs[:,self.opt.nz:2*self.opt.nz]
            negs2 = negs[:,2*self.opt.nz:3*self.opt.nz]
            negs3 = negs[:,3*self.opt.nz:]
            noise3_ = torch.cat([query3, pos3, negs3],0)
            noise2_ = torch.cat([query2, pos2, negs2],0)
            noise1_ = torch.cat([query1, pos1, negs1],0)
            noise0_ = torch.cat([query0, pos0, negs0],0)

            z_conc3 = torch.cat([noise3, noise3_], 0)
            z_conc2 = torch.cat([noise2, noise2_], 0)
            z_conc1 = torch.cat([noise1, noise1_], 0)
            z_conc0 = torch.cat([noise0, noise0_], 0)
            attn_conc = torch.cat([input_attv] * (self.opt.num_negative + 3), 0)
            input_resv3_conc = torch.cat([input_resv3] * (self.opt.num_negative + 3), 0)
            input_resv2_conc = torch.cat([input_resv2] * (self.opt.num_negative + 3), 0)
            input_resv1_conc = torch.cat([input_resv1] * (self.opt.num_negative + 3), 0)

            fake3 = self.netG3(z_conc3, attn_conc)
            fake2 = self.netG2(z_conc2, attn_conc, input_resv3_conc)
            fake1 = self.netG1(z_conc1, attn_conc, input_resv2_conc)
            fake0 = self.netG0(z_conc0, attn_conc, input_resv1_conc)

            fake3_a = fake3[:input_resv3.size(0)]
            fake2_a = fake2[:input_resv2.size(0)]
            fake1_a = fake1[:input_resv1.size(0)]
            fake0_a = fake0[:input_resv0.size(0)]
            fake3_b = fake3[input_resv3.size(0):]
            fake2_b = fake2[input_resv2.size(0):]
            fake1_b = fake1[input_resv1.size(0):]
            fake0_b = fake0[input_resv0.size(0):]

            criticG3_fake = self.netD3(fake3_a, input_attv)
            criticG2_fake = self.netD2(fake2_a, input_attv)
            criticG1_fake = self.netD1(fake1_a, input_attv)
            criticG0_fake = self.netD0(fake0_a, input_attv)
            G3_cost = criticG3_fake.mean()
            G2_cost = criticG2_fake.mean()
            G1_cost = criticG1_fake.mean()
            G0_cost = criticG0_fake.mean()
            G_cost_aver = (G3_cost.item() + G2_cost.item() + G1_cost.item() + G0_cost.item())/4
            criticG3_fake2 = self.netD3(fake3_b[:input_resv3.size(0)], input_attv)
            criticG2_fake2 = self.netD2(fake2_b[:input_resv2.size(0)], input_attv)
            criticG1_fake2 = self.netD1(fake1_b[:input_resv1.size(0)], input_attv)
            criticG0_fake2 = self.netD0(fake0_b[:input_resv0.size(0)], input_attv)
            G3_cost2 = criticG3_fake2.mean()
            G2_cost2 = criticG2_fake2.mean()
            G1_cost2 = criticG1_fake2.mean()
            G0_cost2 = criticG0_fake2.mean()

            input_res3_norm_2 = F.normalize((input_resv3),dim=1)
            input_res2_norm_2 = F.normalize((input_resv2),dim=1)
            input_res1_norm_2 = F.normalize((input_resv1),dim=1)
            input_res0_norm_2 = F.normalize((input_resv0),dim=1)
            fake_res3_a = F.normalize((fake3_a), dim=1)
            fake_res2_a = F.normalize((fake2_a), dim=1)
            fake_res1_a = F.normalize((fake1_a), dim=1)
            fake_res0_a = F.normalize((fake0_a), dim=1)
            fake_res3_b = F.normalize((fake3_b[:input_resv3.size(0)]), dim=1)
            fake_res2_b = F.normalize((fake2_b[:input_resv2.size(0)]), dim=1)
            fake_res1_b = F.normalize((fake1_b[:input_resv1.size(0)]), dim=1)
            fake_res0_b = F.normalize((fake0_b[:input_resv0.size(0)]), dim=1)
            all_features3 = torch.cat((fake_res3_a, fake_res3_b, input_res3_norm_2.detach()), dim=0)
            all_features2 = torch.cat((fake_res2_a, fake_res2_b, input_res2_norm_2.detach()), dim=0)
            all_features1 = torch.cat((fake_res1_a, fake_res1_b, input_res1_norm_2.detach()), dim=0)
            all_features0 = torch.cat((fake_res0_a, fake_res0_b, input_res0_norm_2.detach()), dim=0)
            fake_inter_contras_loss3 = self.inter_contras_criterion(all_features3,
                                                        torch.cat((input_label, input_label, input_label),
                                                                    dim=0))*self.opt.alpha_main
            fake_inter_contras_loss2 = self.inter_contras_criterion(all_features2,
                                                        torch.cat((input_label, input_label, input_label),
                                                                    dim=0))*self.opt.alpha_main
            fake_inter_contras_loss1 = self.inter_contras_criterion(all_features1,
                                                        torch.cat((input_label, input_label, input_label),
                                                                    dim=0))*self.opt.alpha_main
            fake_inter_contras_loss0 = self.inter_contras_criterion(all_features0,
                                                        torch.cat((input_label, input_label, input_label),
                                                                    dim=0))*self.opt.alpha_main
            
            loss_contra3 = 0.0
            loss_contra2 = 0.0
            loss_contra1 = 0.0
            loss_contra0 = 0.0
            for j in range(input_res3.size(0)):
                logits = fake3_b[j:fake3_b.shape[0]:input_res3.size(0)].view(self.opt.num_negative + 2, -1)
                if self.opt.featnorm:
                    logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)
                loss_contra3 += self.compute_contrastive_loss(logits[0:1], logits[1:])
            for j in range(input_res2.size(0)):
                logits = fake2_b[j:fake2_b.shape[0]:input_res2.size(0)].view(self.opt.num_negative + 2, -1)
                if self.opt.featnorm:
                    logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)
                loss_contra2 += self.compute_contrastive_loss(logits[0:1], logits[1:])
            for j in range(input_res1.size(0)):
                logits = fake1_b[j:fake1_b.shape[0]:input_res1.size(0)].view(self.opt.num_negative + 2, -1)
                if self.opt.featnorm:
                    logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)
                loss_contra1 += self.compute_contrastive_loss(logits[0:1], logits[1:])
            for j in range(input_res0.size(0)):
                logits = fake0_b[j:fake0_b.shape[0]:input_res0.size(0)].view(self.opt.num_negative + 2, -1)
                if self.opt.featnorm:
                    logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)
                loss_contra0 += self.compute_contrastive_loss(logits[0:1], logits[1:])
            loss_lz3 = self.opt.alpha_div * loss_contra3
            loss_lz2 = self.opt.alpha_div * loss_contra2
            loss_lz1 = self.opt.alpha_div * loss_contra1
            loss_lz0 = self.opt.alpha_div * loss_contra0

            c_errG = self.cls_criterion(self.classifier(feats=torch.cat((fake0_a, fake1_a, fake2_a, fake3_a), dim=1), classifier_only=True), Variable(input_label))
            c_errG *= self.opt.cls_weight
            c_errG.backward(retain_graph=True)

            scale01_loss = self.multiscale_contras_criterion(torch.cat((fake_res0_a,fake_res1_a),dim=0),
                                                             torch.cat((input_label,input_label),dim=0))*self.opt.alpha_con
            scale02_loss = self.multiscale_contras_criterion(torch.cat((fake_res0_a,fake_res2_a),dim=0),
                                                             torch.cat((input_label,input_label),dim=0))*self.opt.alpha_con
            scale03_loss = self.multiscale_contras_criterion(torch.cat((fake_res0_a,fake_res3_a),dim=0),
                                                             torch.cat((input_label,input_label),dim=0))*self.opt.alpha_con
            scale12_loss = self.multiscale_contras_criterion(torch.cat((fake_res1_a,fake_res2_a),dim=0),
                                                             torch.cat((input_label,input_label),dim=0))*self.opt.alpha_con
            scale13_loss = self.multiscale_contras_criterion(torch.cat((fake_res1_a,fake_res3_a),dim=0),
                                                             torch.cat((input_label,input_label),dim=0))*self.opt.alpha_con
            scale23_loss = self.multiscale_contras_criterion(torch.cat((fake_res2_a,fake_res3_a),dim=0),
                                                             torch.cat((input_label,input_label),dim=0))*self.opt.alpha_con
            scale_contras_loss = (scale01_loss + scale02_loss + scale03_loss + scale12_loss + scale13_loss + scale23_loss)/6
            scale_contras_loss.backward(retain_graph=True)
            
            # Total loss 
            errG3 = -G3_cost - G3_cost2 + loss_lz3 + fake_inter_contras_loss3
            errG2 = -G2_cost - G2_cost2 + loss_lz2 + fake_inter_contras_loss2
            errG1 = -G1_cost - G1_cost2 + loss_lz1 + fake_inter_contras_loss1
            errG0 = -G0_cost - G0_cost2 + loss_lz0 + fake_inter_contras_loss0

            errG3.backward(retain_graph=True)
            errG2.backward(retain_graph=True)
            errG1.backward(retain_graph=True)
            errG0.backward()

            self.optimizerG3.step()
            self.optimizerG2.step()
            self.optimizerG1.step()
            self.optimizerG0.step()

            errG_aver = (errG3.item() + errG2.item() + errG1.item() + errG0.item())/4 + c_errG.item() + scale_contras_loss.item()

            if i%1000==0:
                print(f"{self.gen_type} [{self.epoch+1:02}/{self.opt.nepoch:02}] [{i:06}/{int(self.ntrain)}] \
                Loss: {errG_aver :0.4f} D loss: {D_cost_aver:.4f} G loss: {G_cost_aver:.4f}")
        
        # set to evaluation mode
        self.netG3.eval()
        self.netG2.eval()
        self.netG1.eval()
        self.netG0.eval()
