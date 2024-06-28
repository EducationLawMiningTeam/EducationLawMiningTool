import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def score_model(preds, labels, stageName, use_zero=False):
    mae = np.mean(np.absolute(preds - labels))
    corr = np.corrcoef(preds, labels)[0][1]
    non_zeros = np.array(
        [i for i, e in enumerate(labels) if e != 0 or use_zero])
    preds = preds[non_zeros]
    labels = labels[non_zeros]
    preds1 = preds
    labels1 = labels
    preds = preds >= 0
    labels = labels >= 0
    # indexList = []
    # indexListcorr = []
    # if(stageName != "train") :
    #     print("stage:{}".format(stageName))
    #     for i in range(len(preds)) :
    #         if(preds[i]!=labels[i]) :
    #             indexList.append(i)
    #         else :
    #             indexListcorr.append(i)
    #     for b in range(len(indexList)) :
    #         print("pred:{:.3f}, but label:{:.3f}".format(preds1[indexList[b]], labels1[indexList[b]]))
    #     for c in range(len(indexListcorr)) :
    #         print("pred:{:.3f}, and label:{:.3f}".format(preds1[indexListcorr[c]], labels1[indexListcorr[c]]))
    f_score = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return acc, mae, corr, f_score