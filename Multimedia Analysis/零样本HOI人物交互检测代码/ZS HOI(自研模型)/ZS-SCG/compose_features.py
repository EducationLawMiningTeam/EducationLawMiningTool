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

    hoi_feature = torch.from_numpy(np.load(f'{args.output_root}/hoi_feats.npy')).cuda().to(torch.float32)
    hoi_label = torch.from_numpy(np.load(f'{args.output_root}/hoi_labels.npy')).cuda()
    hoi_obj_label = torch.from_numpy(np.load(f'{args.output_root}/hoi_objs.npy')).cuda()
    hoi_action_label = torch.from_numpy(np.load(f'{args.output_root}/hoi_actions.npy')).cuda()

    human_feature = torch.from_numpy(np.load(f'{args.output_root}/human_feature.npy')).cuda().to(torch.float32)
    obj_feature = torch.from_numpy(np.load(f'{args.output_root}/obj_feature.npy')).cuda().to(torch.float32)
    spatial_feature = torch.from_numpy(np.load(f'{args.output_root}/spatial_feature.npy')).cuda().to(torch.float32)

    # global_feature = torch.zeros(1,256).cuda().to(torch.float32)
    global_feature = torch.load('global_features_example').cuda()

    hoi_feats_compose = []
    for i in range(len(human_feature)):
        human = human_feature[i]
        obj = obj_feature[i]
        spatial = spatial_feature[i]
        # spatial = spatial.expand(1, 1024)
        obj_label = hoi_obj_label[i]
        hoi_feature1 = hoi_feature[i]

        hoi_feature2 = net.interaction_head.box_pair_head.forward_for_compose_features(global_feature, human, obj, spatial, obj_label)
        hoi_feats_compose.append(hoi_feature2[0].cpu().numpy()[0].tolist())
        pass
    
    np.save(f'{args.output_root}/hoi_feats_compose.npy', np.array(hoi_feats_compose))

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
    parser.add_argument('--zero-shot-type', default='nf_uc', type=str)
    
    args = parser.parse_args()
    print(args)
    main(args)


