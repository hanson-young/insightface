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
from tqdm import tqdm
import numba as nb

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc, tp, fp, fn, tn

@nb.njit
def clc_sim(embedding1, embedding2):
    num = np.dot(embedding1, embedding2)
    denom = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    cos = num / denom
    return cos


def clc_cosine(bin, pairs_list):
    cosine = []
    for item in tqdm(pairs_list):
        # if(idx > 100):
        #     break
        tmp = item
        item = item.strip('\n').split('\t')
        index1 = bin[1].index(item[0])
        index2 = bin[1].index(item[1])

        embedding1 = np.array(bin[0][index1])
        embedding2 = np.array(bin[0][index2])
        # num = np.dot(embedding1, embedding2)
        # denom = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        # cos = num / denom
        cos = clc_sim(embedding1, embedding2)
        cosine.append([int(item[2]), cos])
    return cosine


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='/media/yj/hanson/face-recognition/HSVD/clearn-face0816-112x112/bin', help='')
    parser.add_argument('--pairs-path', default='hsvd_pairs.txt', help='meglass pairs path')
    parser.add_argument('--embedding-bin', default='embedding_h3.bin', help='embedding bin file path.')
    # h1 old @0.50
    # ('tpr:%d,%f,%d', 450, 0.07874015748031496, 5715)
    # ('fpr:%d,%f,%d', 4934, 0.00019332666446460778, 25521570)
    # h1 old @0.54
    # ('tpr:%d,%f,%d', 622, 0.10883639545056868, 5715)
    # ('fpr:%d,%f,%d', 1750, 6.856944929328407e-05, 25521570)
    # h2 new1 @0.45
    # ('tpr:%d,%f,%d', 426, 0.07454068241469816, 5715)
    # ('fpr:%d,%f,%d', 822, 3.220804989661686e-05, 25521570)
    # h3 new2@0.46
    # ('tpr:%d,%f,%d', 424, 0.0741907261592301, 5715)
    # ('fpr:%d,%f,%d', 390, 1.528119155678902e-05, 25521570)
    # ('tpr:%d,%f,%d', 427, 0.0747156605424322, 5715)
    # ('fpr:%d,%f,%d', 391, 1.5320374099242326e-05, 25521570)
    # h3 new2@0.50
    # ('tpr:%d,%f,%d', 607, 0.10621172353455818, 5715)
    # ('fpr:%d,%f,%d', 74, 2.8995081415445835e-06, 25521570)
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
    cosine_pool = []
    # f1 = open("../eval/view.txt", 'wb')
    import multiprocessing, math

    m_pools = 32
    n_step = int(math.ceil(len(pairs_lists) / float(m_pools)))
    pool = multiprocessing.Pool(processes=32)
    for i in tqdm(xrange(0, len(pairs_lists), n_step)):
        cosine_pool.append(pool.apply_async(clc_cosine, args=(bin, pairs_lists[i: i + n_step])))
        # print (i, len(cosine_pool[0].get()))

    cosine = []
    for i in range(len(cosine_pool)):
        cosine += cosine_pool[i].get()
    pool.close()
    pool.join()
    # f1 = open("../eval/view.txt", 'wb')
    thresh_lists = np.linspace(0.40, 0.60, 21)
    cosine = np.array(cosine)
    # roc_pool = []
    # pool = multiprocessing.Pool(processes=11)
    for i in xrange(0,thresh_lists.shape[0]):
        tpr, fpr, acc, tp, fp, fn, tn = calculate_accuracy(thresh_lists[i], cosine[:,1], cosine[:,0])
        print("tpr@%f:%f,%d,%d" % (thresh_lists[i], tpr, int(tp), int(fn)))
        print("fpr@%f:%f,%d,%d" % (thresh_lists[i], fpr, int(fp), int(tn)))

    # tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    # fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    # roc = []
    # for i in range(len(roc_pool)):
    #     roc += roc_pool[i].get()
    # pool.close()
    # pool.join()
    #
    # for itx in roc:
    #     thresh = itx[0]
    #     countt = itx[1]
    #     tpr = itx[2]
    #     totalt = itx[3]
    #
    #     countf = itx[4]
    #     fpr = itx[5]
    #     totalf = itx[6]
    #
    #     if totalt != 0:
    #         print("tpr@%f:%d,%d,%f" % (thresh, countt, totalt, tpr))  # 本人没识别出来
    #     if totalf != 0:
    #         print("fpr@%f:%d,%d,%f" % (thresh, countf, totalf, fpr))  # 识别出错的概率
    #
