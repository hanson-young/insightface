"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import numpy as np
import sklearn
import datetime
import pickle
import mxnet as mx
import time
from mxnet import ndarray as nd
sys.path.append(os.path.dirname(__file__))



def load_bin(path, image_size):
    bins, name_list = pickle.load(open(path, 'rb'))
    data_list = []
    # for flip in [0, 1]:
    data_list = nd.empty((len(name_list), 3, image_size[0], image_size[1]))
    # data_list.append(data)
    for i in xrange(len(name_list)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        img = nd.transpose(img, axes=(2, 0, 1))
        # for flip in [0, 1]:
        #     if flip == 1:
        #         img = mx.ndarray.flip(data=img, axis=2)
        data_list[i][:] = img
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, name_list

def embedding(data_set, mx_model, batch_size, label_shape=None, norm=True):
    assert len(data_set) == 2
    data_list = data_set[0]
    name_lists = data_set[1]
    model = mx_model
    embeddings_list = []
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size,))
    else:
        _label = nd.ones(label_shape)
    # for idx in xrange(len(data_list)):
    data = data_list
    embeddings = None
    begin = 0
    while begin < data.shape[0]:
        end = min(begin+batch_size, data.shape[0])
        count = end - begin
        _data = nd.slice_axis(data, axis=0, begin=end-batch_size, end=end)
        db = mx.io.DataBatch(data=(_data,), label=(_label,))
        model.forward(db, is_train=False)
        net_out = model.get_outputs()
        _embeddings = net_out[0].asnumpy()
        if embeddings is None:
            embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
        embeddings[begin:end, :] = _embeddings[(batch_size-count):, :]
        begin = end
    embeddings_list.append(embeddings)

    _xnorm = np.linalg.norm(np.vstack(embeddings_list), axis=1)
    _xnorm = np.mean(_xnorm)

    embeddings = embeddings_list[0]
    if norm == True:
        embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print(embeddings)
    print('infer time', time_consumed)

    return embeddings, name_lists
    # acc_list = cal_acc(embeddings[0::2], embeddings[1::2], issame_list, fold_num=nfolds)
    # acc, std = np.mean(acc_list), np.std(acc_list)
    # if not is_roc:
    #     return acc, std, _xnorm
    # fpr, tpr, thresh, auc_score = cal_roc(embeddings[0::2], embeddings[1::2], issame_list)
    # return acc, std, _xnorm, fpr, tpr, thresh, auc_score

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
    import face_image

    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='/media/handsome/data2/DataSet/FaceVerification/bin/', help='')
    parser.add_argument('--model', default='../../model/model,0',
                        help='path to load model.')
    parser.add_argument('--target', default='MeGlass', help='test targets.')
    parser.add_argument('--embedding-bin', default='embedding.bin', help='test targets.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=24, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    args = parser.parse_args()

    # prop = face_image.load_property(args.data_dir)
    # image_size = prop.image_size
    image_size = (112,112)
    print('image_size', image_size)
    ctx = mx.gpu(args.gpu)

    vec = args.model.split(',')
    print('loading', vec)

    time0 = time.time()
    sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    diff = time.time() - time0
    print('model loading time', diff)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(args.data_dir, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)

    embeddings = None
    name_lists = None
    for i in xrange(len(ver_list)):
        embeddings, name_lists = embedding(ver_list[i], model, args.batch_size)

    embedding_outpath = os.path.join(args.data_dir, args.embedding_bin)
    with open(embedding_outpath, 'wb') as f:
        pickle.dump((embeddings, name_lists), f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(embedding_outpath, 'r') as f:
    #     bin = pickle.load(f)
    #     print(bin)
        # results = []
        # acc, std, _xnorm, fpr, tpr, thresh, auc_score = \
        #     test(ver_list[i], model, args.batch_size, args.nfolds, True)
        # print('[%s]XNorm: %f' % (ver_name_list[i], _xnorm))
        # print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc, std))
        # print('[%s]TPR: %1.5f @ 0.01' % (ver_name_list[i], np.interp(0.01, fpr, tpr)))
        # print('[%s]TPR: %1.5f @ 0.001' % (ver_name_list[i], np.interp(0.001, fpr, tpr)))
        # print('[%s]ROC: %1.5f' % (ver_name_list[i], auc_score))