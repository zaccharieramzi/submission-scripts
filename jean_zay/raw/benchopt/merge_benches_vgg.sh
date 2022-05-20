#!/bin/bash
module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH

cd $WORK/benchmark_resnet_classif

for random_state in {42,}; do
    BASIC_CMD="benchopt run ."
    BASIC_CMD="$BASIC_CMD -o *16 -d cifar[*,random_state=${random_state},with_validation=False] -r 1 -n 200 --timeout 10800"
    args=''
    model="adam-torch[batch_size=128"
    args="${args} -s ${model},coupled_weight_decay=0.0,data_aug=True,decoupled_weight_decay=0.02,*,lr_schedule=step]"

    # SGD Torch
    model="sgd-torch[batch_size=128"
    # vanilla sgd
    args="${args} -s ${model},data_aug=False,*,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]"
    # sgd with data aug
    args="${args} -s ${model},data_aug=True,*,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]"
    # sgd with data aug + momentum
    args="${args} -s ${model},data_aug=True,*,lr_schedule=None,momentum=0.9,nesterov=False,weight_decay=0.0]"
    # sgd with data aug + momentum + cosine
    args="${args} -s ${model},data_aug=True,*,lr_schedule=step,momentum=0.9,nesterov=False,weight_decay=0.0]"
    # sgd with data aug + momentum + step + wd
    args="${args} -s ${model},data_aug=True,*,lr_schedule=step,momentum=0.9,nesterov=False,weight_decay=0.0005]"
done
$BASIC_CMD $args
