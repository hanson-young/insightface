#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/media/handsome/backupdata1/hanson/face-recognition/faces_emore

NETWORK=r50
JOB=softmax1e3
MODELDIR="/media/handsome/backupdata1/hanson/face-recognition/models/model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 0 --prefix "$PREFIX" --per-batch-size 96

#> "$LOGFILE" 2>&1 &