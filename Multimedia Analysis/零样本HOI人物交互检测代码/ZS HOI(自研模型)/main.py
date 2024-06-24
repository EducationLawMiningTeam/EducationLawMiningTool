# from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
from hoi_arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
from hoi_dataset import FeaturesCls, FeaturesGAN

from hoi_train_gan_contra import TrainGAN
from generate import load_unseen_att, load_all_att
from mmdetection.splits import get_unseen_class_labels
from rare_and_no import get_rare
from index2some import hoi2obj

opt = parse_args()

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

try:
    os.makedirs(opt.outname)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

for arg in vars(opt): print(f"######################  {arg}: {getattr(opt, arg)}")


print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)

torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


attributes, lables = load_all_att(opt)
ids = unseen
unseen_attributes, unseen_att_labels = attributes[ids], lables[ids]
unseenLabels_obj = [hoi2obj(label) for label in unseen_att_labels]
# unseen_attributes, unseen_att_labels = load_unseen_att(opt)






seenDataset = FeaturesGAN(opt)
trainFGGAN = TrainGAN(opt, attributes, unseen_attributes, unseen_att_labels, seen_feats_mean=seenDataset.features_mean, gen_type='FG')

start_epoch = 199

if start_epoch != 0:
    trainFGGAN.netG = torch.load(f'./results_{opt.zero_shot_type}_{opt.use_type}/G_{opt.batch_size}_{start_epoch}_{opt.zero_shot_type}.pth')
    trainFGGAN.netD = torch.load(f'./results_{opt.zero_shot_type}_{opt.use_type}/D_{opt.batch_size}_{start_epoch}_{opt.zero_shot_type}.pth')

# if opt.netD and opt.netG:
#     start_epoch = trainFGGAN.load_checkpoint()

for epoch in range(start_epoch, 300):
    # features, labels = seenDataset.epochData(include_bg=False)
    features, labels, actions, objs = seenDataset.epochData(include_bg=False)  # 一个疑问，这里取了多少？？ 答：就是全部
    # train GAN
    trainFGGAN(epoch, features, labels, actions, objs)

    if (epoch + 1) % 5 == 0:
        torch.save(trainFGGAN.netG, f'./results_{opt.zero_shot_type}_{opt.use_type}/G_{opt.batch_size}_{epoch}_{opt.zero_shot_type}.pth')
        torch.save(trainFGGAN.netD, f'./results_{opt.zero_shot_type}_{opt.use_type}/D_{opt.batch_size}_{epoch}_{opt.zero_shot_type}.pth')
        print(111111111111111111111111)
        
    # # synthesize features
    # syn_feature, syn_label, syn_obj = trainFGGAN.generate_syn_feature(unseen_att_labels, unseen_attributes, unseenLabels_obj, num=opt.syn_num)
    # # num_of_bg = opt.syn_num*2   这里是background信息，不会用到


"""
opt中需要更换的部分


CUDA_VISIBLE_DEVICES=6 nohup python main.py --zero_shot_type nf_uc --cuda > data_nf_uc/output.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python main.py --zero_shot_type ua --cuda > data_ua/output_c.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python main.py --zero_shot_type ua --use_type con --cuda > data_ua/output_c.log 2>&1 &
"""
