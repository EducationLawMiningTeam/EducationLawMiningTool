from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
from dataset import FeaturesCls, FeaturesGAN
from train_cls import TrainCls
from train_gan import TrainGAN
from generate import load_unseen_att, load_all_att
from mmdetection.splits import get_unseen_class_labels


opt = parse_args()


try:
    os.makedirs(opt.outname)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

opt.num_negative=10
opt.radius=0.0000001
opt.alpha_div=0.001 # alpha intra-scale diverge
opt.alpha_main=0.001  # alpha intra-scale maintain
opt.alpha_con=0.001 # alpha inter-scale construct
opt.tau=0.1 # temperature

for arg in vars(opt): print(f"######################  {arg}: {getattr(opt, arg)}")


print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)

torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

unseen_attributes, unseen_att_labels = load_unseen_att(opt)
attributes, _ = load_all_att(opt)
# init classifier
trainCls = TrainCls(opt)

print('initializing GAN Trainer')


start_epoch = 0

load_path = "checkpoints/coco6515"
resume_train = 0    #0 for no; 1 for best; 2 for latest

seenDataset = FeaturesGAN(opt)
trainFGGAN = TrainGAN(opt, attributes, unseen_attributes, unseen_att_labels, gen_type='FG')

if resume_train != 0:
    trainCls.load_checkpoint(load_path, resume_train)
    start_epoch = trainFGGAN.load_checkpoint(load_path, resume_train)

for epoch in range(start_epoch, opt.nepoch):

    features, labels = seenDataset.epochData(include_bg=True)
    
    # train GAN
    trainFGGAN(epoch, features, labels) 
    del features, labels
    # synthesize features
    syn_feature, syn_label = trainFGGAN.generate_syn_feature(unseen_att_labels, unseen_attributes, num=opt.syn_num)

    num_of_bg = opt.syn_num*2

    real_feature_bg, real_label_bg = seenDataset.getBGfeats(num_of_bg)

    
    # concatenate synthesized + real bg features
    syn_feature = np.concatenate((syn_feature.data.numpy(), real_feature_bg))
    syn_label = np.concatenate((syn_label.data.numpy(), real_label_bg))
    
    trainCls(syn_feature, syn_label, gan_epoch=epoch)
    del syn_feature, syn_label
    # -----------------------------------------------------------------------------------------------------------------------
    # plots
    classes = np.concatenate((['background'], get_unseen_class_labels(opt.dataset, split=opt.classes_split)))
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Train.npy'), classes, classes, opt, dataset='Train', prefix=opt.class_embedding.split('/')[-1])
    plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Test.npy'), classes, classes, opt, dataset='Test', prefix=opt.class_embedding.split('/')[-1])
    plot_acc(np.vstack(trainCls.val_accuracies), opt, prefix=opt.class_embedding.split('/')[-1])

    # save models
    if trainCls.isBestIter == True:
        trainFGGAN.save_checkpoint(state='best')

    trainFGGAN.save_checkpoint(state='latest')