#encoding=utf-8
import pickle
import sys
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import KFold
from scipy import interp
import matplotlib.pyplot as plt

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def find_best_acc(labels, scores):
    best_thresh = None
    best_acc = 0
    thresholds = np.arange(0, 1.0, 0.01)
    for thresh in thresholds:
        preds = np.greater_equal(scores, thresh).astype(np.int32)
        acc = accuracy_score(labels, preds, normalize=True)
        if thresh == 0.46:
            print(thresh, acc)
        if acc > best_acc:
            best_thresh = thresh
            best_acc = acc
    return best_acc, best_thresh

def cal_acc(fold_num=10):

    scores = cosine[:,1]
    labels = cosine[:,0]
    indices = np.arange(scores.shape[0])
    k_fold = KFold(n_splits=fold_num)

    acc_list = []
    for train_set, test_set in k_fold.split(indices):
        _, best_thresh = find_best_acc(labels[train_set], scores[train_set])

        test_preds = np.greater_equal(scores[test_set], best_thresh).astype(np.int32)
        print(best_thresh)
        acc_list.append(accuracy_score(labels[test_set], test_preds, normalize=True))
    return acc_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='/media/handsome/data2/DataSet/FaceVerification/HSVD/clearn-face0816-112x112/bin/', help='')
    parser.add_argument('--pairs-path', default='hsvd_pairs.txt', help='meglass pairs path')
    parser.add_argument('--embedding-bin', default='embedding.bin', help='embedding bin file path.')

    args = parser.parse_args()

    embedding_path = os.path.join(args.data_dir, args.embedding_bin)
    pairs_path = os.path.join(args.data_dir, args.pairs_path)

    # paris_lists = None
    with open(pairs_path,'r') as f:
        pairs_lists = f.readlines()

    # print(pairs_lists[0:300])
    with open(embedding_path, 'r') as f:
        bin = pickle.load(f)
        # print(bin)

    f1 = open("../eval/view.txt",'wb')
    cosine = []
    countt = 0
    countf = 0

    totalt = 0
    totalf = 0
    for idx, item in enumerate(pairs_lists):
        # if(idx > 100):
        #     break
        tmp = item
        item = item.strip('\n').split('\t')
        index1 = bin[1].index(item[0])
        index2 = bin[1].index(item[1])

        embedding1 = np.array(bin[0][index1])
        embedding2 = np.array(bin[0][index2])
        num = np.dot(embedding1, embedding2)
        denom = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        cos = num / denom
        cosine.append([int(item[2]),cos])
        if int(item[2]) == 1:
            totalt += 1
        if int(item[2]) == 0:
            totalf += 1
        if cos < 0.46 and int(item[2]) == 1:#best threshold is 0.27 in flw
            countt += 1


            # print(item, cos)

        if cos > 0.46 and int(item[2]) == 0:
            f1.write(tmp)
            countf += 1
        # print(cos)
        # if(cos > 0.5):
        # print(item)
    f1.close()
    print("tpr:%d,%f,%d",countt,float(countt) / totalt,totalt) #本人没识别出来
    print("fpr:%d,%f,%d", countf,float(countf) / totalf,totalf) #识别出错的概率
    cosine = np.array(cosine)

    acc_list = cal_acc()
    print(acc_list)
    acc, std = np.mean(acc_list), np.std(acc_list)

    print('[%s]Accuracy: %1.5f+-%1.5f' % ("hsvd", acc, std))


    # drawing roc curve
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100000)
    all_tpr = []

    # print((result[:,0], result[:,1]))
    fpr, tpr, thresholds = roc_curve(cosine[:,0], cosine[:,1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (0, roc_auc))


    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= 1
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # print mean_fpr,len(mean_fpr)
    print (mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
