"""
Visualise box pairs in an image

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

# 这是第一版代码v1，没有考虑多标签的问题(也不需要考虑)

import os
import sys
import torch
import pocket
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
from tqdm import tqdm
import torchvision.ops.boxes as box_ops
from index2some import *

from ops import compute_spatial_encodings, binary_focal_loss

sys.path.append('/'.join(os.path.abspath(sys.argv[0]).split('/')[:-2]))

from utils import custom_collate, DataFactory
from models import SpatiallyConditionedGraph as SCG

box_nms_thresh = 0.5

def colour_pool(n):
    pool = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#17becf', '#e377c2'
    ]
    nc = len(pool)

    repeat = n // nc
    big_pool = []
    for _ in range(repeat):
        big_pool += pool
    return big_pool + pool[:n%nc]


def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()

def visualise_entire_image(dataset, output, i):
    """Visualise bounding box pairs in the whole image by classes"""
    bh=output['boxes_h']
    bo=output['boxes_o']
    no = len(bo)

    bbox, inverse = torch.unique(torch.cat([bo, bh]), dim=0, return_inverse=True)
    idxh = inverse[no:]
    idxo = inverse[:no]

    im = dataset.dataset.load_image(
        os.path.join(
            dataset.dataset._root,
            dataset.dataset.filename(i)
        )
    )

    # Print predicted classes and scores
    scores = output['scores']
    prior = output['prior']
    index = output['index']
    pred = output['prediction']
    labels = output['labels']
    
    unique_actions = torch.unique(pred)
    for verb in unique_actions:
        print(f"\n=> Action: {dataset.dataset.verbs[verb]}")
        sample_idx = torch.nonzero(pred == verb).squeeze(1)
        for idx in sample_idx:
            b_idx = index[idx]
            print(
                f"({idxh[b_idx].item():<2}, {idxo[b_idx].item():<2}),",
                f"score: {scores[idx]:.4f}, prior: {prior[0, idx]:.2f}, {prior[1, idx]:.2f}",
                f"label: {bool(labels[idx])}"
            )
            if bool(labels[idx]):
                print(f"({idxh[b_idx].item():<2}, {idxo[b_idx].item():<2})", f"=> Action: {dataset.dataset.verbs[verb]}")

    # Draw the bounding boxes
    fig = plt.figure()
    plt.imshow(im)
    ax = plt.gca()
    draw_boxes(ax, bbox)
    # 保存图片
    fig.savefig('../figs/output_fig_{}.png'.format(i))

    # 关闭图形窗口
    plt.close(fig)


@torch.no_grad()
def main(args):
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    dataset = DataFactory(
        name='hicodet', partition=args.partition,
        data_root=args.data_root,
        detection_root=args.detection_dir,
        zero_shot_type=args.zero_shot_type
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=custom_collate,
        batch_size=4, shuffle=False
    )

    net = SCG(
        dataset.dataset.object_to_verb, 49,
        num_iterations=args.num_iter,
        box_score_thresh=args.box_score_thresh
    )

    net.cuda()
    net.eval()

    if os.path.exists(args.model_path):
        print("\nLoading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
            "Proceed to use a randomly initialised model.\n")
    else:
        print("\nProceed with a randomly initialised model\n")

    hoi_feature = []
    hoi_human_feature = []
    hoi_obj_feature = []
    hoi_spatial_feature = []

    hoi_label = []
    hoi_obj_label = []
    hoi_action_label = []
    start_image = 0

    if start_image != 0:
        hoi_feature = np.load(f'{args.output_root}/hoi_feats.npy')
        hoi_label = np.load(f'{args.output_root}/hoi_labels.npy').tolist()
        hoi_obj_label = np.load(f'{args.output_root}/hoi_action_labels.npy').tolist()
        hoi_action_label = np.load(f'{args.output_root}/hoi_obj_labels.npy')

    # torch.save(net.interaction_head.box_pair_predictor, f'{args.output_root}/box_pair_predictor.pth')
    # torch.save(net.interaction_head.box_pair_suppressor, f'{args.output_root}/box_pair_suppressor.pth')


    for i in tqdm(range(len(dataset))):
        if i < start_image:
            continue
        images, detections, targets = pocket.ops.relocate_to_cuda(dataset[i])   # 这里的detection不会被用到

        images = [images]; detections = [detections]; targets = [targets]

        images, detections, targets, original_image_sizes = net.preprocess(
                images, detections, targets)

        features = net.backbone(images.tensors)

        
        for index in range(targets[0]['labels'].shape[0]):   # 遍历目标中的所用gt hoi组合
            box_coords = [torch.cat((targets[0]['boxes_h'][index].unsqueeze(0), targets[0]['boxes_o'][index].unsqueeze(0)),dim=0)]
            box_labels = [torch.cat((torch.tensor([49]).cuda(), targets[0]['object'][index].unsqueeze(0)), dim=0)]
            box_scores = [torch.tensor([1., 1.]).cuda()]


            box_features = net.interaction_head.box_roi_pool(features, box_coords, images.image_sizes)

            box_pair_features, boxes_h, boxes_o, object_class,\
            box_pair_labels, box_pair_prior = net.interaction_head.box_pair_head.forward_for_extract_features(
                features, images.image_sizes, box_features,
                box_coords, box_labels, box_scores, targets, 
            )


            # 这一部分只是用来测试提取的特征正确与否
            # obj= targets[0]['object'][index]
            # verb=targets[0]['verb'][index]
            # verb=np.eye(117)[verb]
            # prior = object2valid_action(obj.cpu())
            # pp = find_indices(prior, 1)
            # print('real:', logits_p[0][pp].cpu().detach().numpy())
            # print(verb[pp])

            hoi_feature.append(box_pair_features[0].cpu().numpy()[0].tolist())
            
            temp = np.zeros(117)
            temp[targets[0]['verb'][index]] = 1
            hoi_action_label.append(temp)

            temp = np.zeros(600)
            temp[targets[0]['hoi'][index]] = 1
            hoi_label.append(temp)

            hoi_obj_label.append( np.reshape(targets[0]['object'][index].cpu().numpy(), (1)) )

            # 这是上一个版本的代码，先留着
            # hoi_action_label.append( np.reshape(targets[0]['verb'][index].cpu().numpy(), (1)) )  
            # hoi_obj_label.append( np.reshape(targets[0]['object'][index].cpu().numpy(), (1)) )
            # hoi_label.append( np.reshape(targets[0]['hoi'][index].cpu().numpy(), (1)) )
            
            # 最新的点（idea）就是提取人和物单独的特征
            box_human_obj_features = net.interaction_head.box_pair_head.box_head(box_features)
            hoi_human_feature.append(box_human_obj_features.cpu().numpy()[0].tolist())
            hoi_obj_feature.append(box_human_obj_features.cpu().numpy()[1].tolist())

            box_spatial_feature = compute_spatial_encodings(
                [box_coords[0][[0]]], [box_coords[0][[1]]], [images.image_sizes[0]]
            )
            box_spatial_feature = net.interaction_head.box_pair_head.spatial_head(box_spatial_feature)

            hoi_spatial_feature.append(box_spatial_feature.cpu().numpy()[0].tolist())

            pass
            
        # print(len(hoi_feature), len(hoi_action_label))
        if (i + 1) % 2 == 0:
            # feature = np.array(hoi_feature)
            # label = np.array(hoi_label).astype(int)
            # obj_label = np.concatenate(hoi_obj_label)
            # action_label = np.array(hoi_action_label).astype(int)

            # human_feature = np.array(hoi_human_feature)
            # obj_feature = np.array(hoi_obj_feature)
            # spatial_feature = np.array(hoi_spatial_feature)

            # print(f"{feature.shape} num of features")
            # print(len(hoi_feature))
            # print('save {}'.format(i))
            # np.save(f'{args.output_root}/hoi_feats.npy', feature)
            # np.save(f'{args.output_root}/hoi_labels.npy', label)
            # np.save(f'{args.output_root}/hoi_objs.npy', obj_label)
            # np.save(f'{args.output_root}/hoi_actions.npy', action_label)

            # np.save(f'{args.output_root}/human_feature.npy', human_feature)
            # np.save(f'{args.output_root}/obj_feature.npy', obj_feature)
            # np.save(f'{args.output_root}/spatial_feature.npy', spatial_feature)
            pass

        if (i + 1) == len(dataset):
            feature = np.array(hoi_feature)
            label = np.array(hoi_label).astype(int)
            obj_label = np.concatenate(hoi_obj_label)
            action_label = np.array(hoi_action_label).astype(int)

            human_feature = np.array(hoi_human_feature)
            obj_feature = np.array(hoi_obj_feature)
            spatial_feature = np.array(hoi_spatial_feature)

            print(f"{feature.shape} num of features")
            print(len(hoi_feature))
            print('save {}'.format(i))
            np.save(f'{args.output_root}/hoi_feats.npy', feature)
            np.save(f'{args.output_root}/hoi_labels.npy', label)
            np.save(f'{args.output_root}/hoi_objs.npy', obj_label)
            np.save(f'{args.output_root}/hoi_actions.npy', action_label)

            np.save(f'{args.output_root}/human_feature.npy', human_feature)
            np.save(f'{args.output_root}/obj_feature.npy', obj_feature)
            np.save(f'{args.output_root}/spatial_feature.npy', spatial_feature)


def find_indices(lst, value):
    indices = []
    for i, x in enumerate(lst):
        if x == value:
            indices.append(i)
    return indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', default='./hicodet', type=str)
    parser.add_argument('--detection-dir', default='./hicodet/detections/train2015_gt',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='train2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='nf_uc/ckpt_08320_08.pt', type=str)
    parser.add_argument('--output-root', default='new_extract_nf_uc', type=str)
    parser.add_argument('--zero-shot-type', default='', type=str)
    
    args = parser.parse_args()
    print(args)
    main(args)

# CUDA_VISIBLE_DEVICES=2 nohup python extract_features.py --zero-shot-type nf_uc &  提取unseen类别的特征
# CUDA_VISIBLE_DEVICES=2 nohup python extract_features.py --zero-shot-type '' &  提取所用的特征
