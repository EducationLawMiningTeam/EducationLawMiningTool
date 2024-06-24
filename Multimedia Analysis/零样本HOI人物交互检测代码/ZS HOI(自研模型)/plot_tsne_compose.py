# from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import torch 
from util import *
from hoi_arguments import parse_args

colors = ['#8a2244', '#da8c22', '#c687d5', '#80d6f8', '#440f06', '#000075', '#000000', '#e6194B', '#f58231', '#ffe119', '#bfef45']
colors2 = ['#02a92c', '#3a3075', '#3dde43', '#baa980', '#170eb8', '#f032e6', '#a9a9a9', '#fabebe', '#ffd8b1', '#fffac8', '#aaffc3']

dataset = 'hoi'

opt = parse_args()


# [39, 370, 457]            s_ua 
# [20, 42, 109]                   u_ua 20:straddle bicycle:2250 42:stand_on boat:1626 109:sit_at dining_table:3209
# [18, 245, 577]           u_nf_uc  18:ride bicycle  245:sit_on bench  577:hold umbrella
# [39, 109, 454]           s_nf_uc 39:row boat    109:sit_at dining_table 380:wield knife:583 454:flip skateboard
# [154, 208, 472]     154:sit_on motorcycle  472:wear skis  208:carry backpack 


labels_to_plot = np.array( [39, 109, 454]  ) 

from index2some import hoi_dict

id_to_class = {idx: ' '.join(class_label) for idx, class_label in enumerate(hoi_dict)}

def tsne_plot_feats(f_feat, f_labels, path_save):
    # import pdb; pdb.set_trace()
    tsne = TSNE(n_components=2, random_state=0, verbose=True)
    syn_feature = np.load(f_feat)
    syn_label = np.load(f_labels)
    
    idx = np.where(np.isin(syn_label, labels_to_plot))[0]
    idx = np.random.permutation(idx)[0:400]   # 原来是4000
    X_sub = syn_feature[idx]
    y_sub = syn_label[idx]
    # targets = np.unique(y)

    # colors = []
    for i in range(len(labels_to_plot)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    print(X_sub.shape, y_sub.shape, labels_to_plot.shape)

    X_2d = tsne.fit_transform(X_sub)
    fig = plt.figure(figsize=(6, 5))
    for i, c in zip(labels_to_plot, colors):
        plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c, label=id_to_class[i])
    plt.legend()
    # plt.show()
    fig.savefig(path_save)
    print(f"saved {path_save}")
    return X_sub, y_sub


def plot_unseen(epochs=20):
    real_f, real_l = tsne_plot_feats('new_extract_nf_uc/hoi_feats.npy', 'new_extract_nf_uc/hoi_labels_label.npy', 'real.png')
    print(f"len of real feats: {len(real_f)}")
    for epoch in range(0, epochs):
        f_feat = 'new_extract_nf_uc/hoi_feats_compose.npy'
        f_labels = 'new_extract_nf_uc/hoi_labels_label.npy'
        path_save = 'compose.png'
        syn_f, syn_l = tsne_plot_feats(f_feat, f_labels, path_save)
        print(f"len of syn feats: {len(syn_f)}")
        
        # merge and plot
        feats_all = np.concatenate((syn_f, real_f))
        label_all = np.concatenate((syn_l, real_l))
        tsne = TSNE(n_components=2, random_state=0, verbose=True)
        print(f"len of all feats: {len(feats_all)}")

        X_2d = tsne.fit_transform(feats_all)

        fig = plt.figure(figsize=(6, 5))
        for i, c1, c2 in zip(labels_to_plot, colors, colors2):
            indx = np.where(label_all == i)[0]
            plt.scatter(X_2d[indx[indx<400],   0], X_2d[indx[indx<400], 1], marker='s', c=c1, label=f"s_{id_to_class[i]}")
            plt.scatter(X_2d[indx[indx>=400], 0], X_2d[indx[indx>=400], 1], marker='^', c=c2, label=f"r_{id_to_class[i]}")
        plt.legend()
        fig.savefig('both.png')
        
        print(f"{epoch:02}/{epochs} ")

        plt.close('all')

plot_unseen(1)


