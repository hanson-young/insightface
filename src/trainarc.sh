#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/home/handsome/Documents/data/faces_emore

NETWORK=y1
JOB=softmax12
MODELDIR="/media/handsome/backupdata1/hanson/face-recognition/models/model-y1-softmax"
MODELDIR1="/media/handsome/backupdata1/hanson/face-recognition/models/model-y1-softmax/1"
MODELDIR2="/media/handsome/backupdata1/hanson/face-recognition/models/model-y1-softmax/2"
MODELDIR3="/media/handsome/backupdata1/hanson/face-recognition/models/model-y1-softmax/3"
MODELDIR4="/media/handsome/backupdata1/hanson/face-recognition/models/model-y1-softmax/4"
mkdir -p "$MODELDIR"
mkdir -p "$MODELDIR1"
mkdir -p "$MODELDIR2"
mkdir -p "$MODELDIR3"
mkdir -p "$MODELDIR4"
PREFIX="$MODELDIR/model-$NETWORK-$JOB"
PREFIX1="$MODELDIR1/model-$NETWORK-$JOB"
PREFIX2="$MODELDIR2/model-$NETWORK-$JOB"
PREFIX3="$MODELDIR3/model-$NETWORK-$JOB"
PREFIX4="$MODELDIR4/model-$NETWORK-$JOB"
#LOGFILE="$MODELDIR/log"

#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network m1 --loss-type 0 --data-dir $DATA_DIR --prefix $MODELDIR
CUDA_VISIBLE_DEVICES='0,1,3' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 0 --lr 0.1  --per-batch-size 224 --emb-size 512 --fc7-wd-mult 10  --data-dir  $DATA_DIR --pretrained $PREFIX,0 --prefix $PREFIX1

CUDA_VISIBLE_DEVICES='0,1,3' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4  --lr 0.01 --lr-steps 40000,60000,70000 --wd 0.00004 --fc7-wd-mult 10 --emb-size 512 --per-batch-size 224 --margin-s 128 --data-dir $DATA_DIR --pretrained $PREFIX1,79 --prefix $PREFIX2


CUDA_VISIBLE_DEVICES='0,1,3' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4  --lr 0.001 --lr-steps 40000,60000,70000 --wd 0.00004 --fc7-wd-mult 10 --emb-size 512 --per-batch-size 224 --margin-s 64 --data-dir $DATA_DIR --pretrained $PREFIX2,46 --prefix $PREFIX3

CUDA_VISIBLE_DEVICES='0,1,3' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4  --lr 0.001 --lr-steps 40000,60000,70000 --wd 0.00004 --fc7-wd-mult 10 --emb-size 512 --per-batch-size 224 --margin-s 64 --data-dir $DATA_DIR --pretrained $PREFIX3,5 --prefix $PREFIX4
#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4  --lr 0.0001  --wd 0.00004 --fc7-wd-mult 10 --emb-size 512 --per-batch-size 150 --margin-s 64 --data-dir ../data/faces_ms1m_112x112 --pretrained ../models/MF/model-y1-arcfaceredodododo,6 --prefix ../models/MF/model-y1-arcfaceredodododod
#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network m1 --loss-type 12 --lr 0.005 --mom 0.0 --per-batch-size 150 --fc7-wd-mult 10 --data-dir ../data/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-softmax,55 --prefix ../models/MobileFaceNet/model-y1-triplet

#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 0 --per-batch-size 150 --emb-size 128 --fc7-wd-mult 10  --data-dir  ../data/faces_ms1m_112x112   --prefix ../models/MobileFaceNet/model-y1-softmax
#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr-steps 120000,180000,210000,230000 --emb-size 128 --per-batch-size 150 --data-dir ../data/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-softmax,20 --prefix ../models/MobileFaceNet/model-y1-arcface

#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr 0.001 --lr-steps 40000,60000,70000 --wd 0.00004 --fc7-wd-mult 10 --emb-size 128 --per-batch-size 150 --margin-s 128 --data-dir ../data/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-arcface,117 --prefix ../models/MobileFaceNet/model-y1-arcface
#


#CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr 0.001  --wd 0.00004 --fc7-wd-mult 10 --emb-size 128 --per-batch-size 150 --margin-s 64 --data-dir ../data/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-arcface,23 --prefix ../models/MobileFaceNet/model-y1-arcface
