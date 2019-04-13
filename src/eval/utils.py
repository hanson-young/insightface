import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import KFold


def cal_l2(emd1, emd2):
    diff = emd1 - emd2
    return -np.sum(np.square(diff), axis=1)


def find_best_acc(labels, scores):
    best_thresh = None
    best_acc = 0
    for score in scores:
        preds = np.greater_equal(scores, score).astype(np.int32)
        acc = accuracy_score(labels, preds, normalize=True)
        if acc > best_acc:
            best_thresh = score
            best_acc = acc
    return best_acc, best_thresh


def cal_acc(embeddings1, embeddings2, labels, fold_num=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0] == len(labels))
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    labels = np.array(labels)

    scores = cal_l2(embeddings1, embeddings2)
    indices = np.arange(embeddings1.shape[0])
    k_fold = KFold(n_splits=fold_num)

    acc_list = []
    for train_set, test_set in k_fold.split(indices):
        _, best_thresh = find_best_acc(labels[train_set], scores[train_set])
        test_preds = np.greater_equal(scores[test_set], best_thresh).astype(np.int32)
        acc_list.append(accuracy_score(labels[test_set], test_preds, normalize=True))

    return acc_list


def cal_roc(embeddings1, embeddings2, labels):
    assert (embeddings1.shape[0] == embeddings2.shape[0] == len(labels))
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    scores = cal_l2(embeddings1, embeddings2)
    labels = np.array(labels)

    fpr, tpr, thresh = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, thresh, auc_score


def plot_roc(fpr, tpr, label):
    plt.plot(fpr, tpr, label=label)
    plt.title('Receiver Operating Characteristics')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.plot([0, 1], [0, 1], 'g--')
    plt.grid(True)
    plt.show()