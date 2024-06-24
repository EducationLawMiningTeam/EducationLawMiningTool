"""
Test a model and compute detection mAP

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader

import pocket

from hicodet.hicodet import HICODet
from models import SpatiallyConditionedGraph as SCG
from utils import DataFactory, custom_collate, test


def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    # num_anno = torch.tensor(HICODet(None, anno_file=os.path.join(
    #     args.data_root, 'instances_train2015.json')).anno_interaction)
    
    # count = 0
    # for i in num_anno:
    #     count = count + i
    # print(count)


    # rare = torch.nonzero(num_anno < 10).squeeze(1)
    # non_rare = torch.nonzero(num_anno >= 10).squeeze(1)

    uv = [4, 6, 12, 15, 18, 25, 34, 38, 40, 49, 58, 60, 68, 69, 72, 73, 77, 82, 96, 97, 104, 113, 116, 118,
                    122, 129, 139, 147,
                    150, 153, 165, 166, 172, 175, 176, 181, 190, 202, 210, 212, 219, 227, 228, 233, 235, 243, 298, 313,
                    315, 320, 326, 336,
                    342, 345, 354, 372, 401, 404, 409, 431, 436, 459, 466, 470, 472, 479, 481, 488, 491, 494, 498, 504,
                    519, 523, 535, 536,
                    541, 544, 562, 565, 569, 572, 591, 595]  # uv

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
                   198, 168, 391, 192, 595, 136, 581]   # rf-uc

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
                383, 93, 516, 64]  # nf-uc

    ua = [
            2, 10, 14, 20, 27, 33, 36, 42, 46, 57, 68, 81, 82, 86, 90, 92, 101, 103,
            109, 111, 116, 120, 121, 122, 123, 136, 137, 138, 140, 141, 149, 152, 155,
            160, 161, 170, 172, 174, 180, 188, 205, 208, 215, 222, 225, 236, 247, 260,
            265, 271, 273, 279, 283, 288, 295, 300, 301, 306, 310, 311, 315, 318, 319,
            337, 344, 352, 356, 363, 369, 373, 374, 419, 425, 427, 438, 453, 458, 461,
            464, 468, 471, 475, 480, 486, 489, 490, 496, 504, 506, 513, 516, 524, 528,
            533, 542, 555, 565, 576, 590, 597
        ]  # ua
    
    uo = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                    294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                    338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                    429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                    537, 558, 559, 560, 561, 595, 596, 597, 598, 599]  # uo

    unseen = []
    if args.zero_shot_type == 'uv':
        unseen = uv
    elif args.zero_shot_type == 'ua':
        unseen = ua
    elif args.zero_shot_type == 'nf_uc':
        unseen = nf_uc
    elif args.zero_shot_type == 'rf_uc':
        unseen = rf_uc
    elif args.zero_shot_type == 'uo':
        unseen = uo
    elif args.zero_shot_type == 'deafult':
        unseen = []
    else:
        return 
    seen = [i for i in range(600)]
    for i in unseen:
        seen.remove(i)
    seen = torch.tensor(seen)
    unseen = torch.tensor(unseen)

    dataloader = DataLoader(
        dataset=DataFactory(
            name='hicodet', partition=args.partition,
            data_root=args.data_root,
            detection_root=args.detection_dir,
        ), collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )

    net = SCG(
        dataloader.dataset.dataset.object_to_verb, 49,
        num_iterations=args.num_iter,
        max_human=args.max_human,
        max_object=args.max_object,
        box_score_thresh=args.box_score_thresh
    )
    
    epoch = 0
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])

        if args.unseen_model != '':
            net.interaction_head.box_pair_predictor = torch.load(args.unseen_model)

        epoch = checkpoint["epoch"]
        
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
            "Proceed to use a randomly initialised model.\n")
    
    

    net.cuda()
    timer = pocket.utils.HandyTimer(maxlen=1)
    
    with timer:
        test_ap = test(net, dataloader)
        if args.save_path != '':
            torch.save(test_ap, args.save_path)

    print("Model at epoch: {} | time elapsed: {:.2f}s\n"
        "Full: {:.4f}, seen: {:.4f}, unseen: {:.4f}".format(
        epoch, timer[0], test_ap.mean(),
        test_ap[seen].mean(),test_ap[unseen].mean()
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--detection-dir', default='hicodet/detections/test2015_finetuned_drg',  # test2015_finetuned_drg
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='nf_uc_no_mask/ckpt_06240_06.pt', type=str)
    parser.add_argument('--unseen-model', default='', type=str)
    parser.add_argument('--zero-shot-type', default='nf_uc', type=str)
    parser.add_argument('--save-path', default='', type=str)
    
    args = parser.parse_args()
    print(args)

    main(args)


# CUDA_VISIBLE_DEVICES=2 nohup python test.py --model-path uv2/ckpt_10374_07.pt --zero-shot-type uv > uv2_7.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python test.py --model-path nf_uc_no_mask/ckpt_06240_06.pt --zero-shot-type nf_uc > nf_uc_no_mask/test6.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python test.py  --model-path ua/ckpt_07945_07.pt --unseen-model /data01/liuchuan/zsd2/results_ua_con/classifier_latest_199_1.pth --zero-shot-type ua > ua/con_199_1.txt 2>&1 & 