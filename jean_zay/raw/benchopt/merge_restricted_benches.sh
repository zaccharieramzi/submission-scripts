#!/bin/bash
module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH

cd $WORK/benchmark_resnet_classif

for dataset in {'cifar', 'svhn'}; do
# for dataset in {'mnist',}; do
    for with_validation in {'True','False'}; do
        BASIC_CMD="benchopt run ."
        BASIC_CMD="$BASIC_CMD -o *18 -d ${dataset}[*,with_validation=${with_validation}] -r 1 -n 200 --timeout 10800"
        args=''
        if [ "${with_validation}" == "True" ]; then
            model="sgd-torch[batch_size=128"
            args="${args} -s ${model},data_aug=True,*,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]"
            args="${args} -s ${model},data_aug=True,*,lr_schedule=step,momentum=0.9,nesterov=False,weight_decay=0.0005]"
        else
            model="adam-torch[batch_size=128"
            args="${args} -s ${model},coupled_weight_decay=0.0,data_aug=True,decoupled_weight_decay=0.02,*,lr_schedule=cosine]"

            # SGD Torch
            model="sgd-torch[batch_size=128"
            # vanilla sgd
            args="${args} -s ${model},data_aug=False,*,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]"
            # sgd with data aug
            args="${args} -s ${model},data_aug=True,*,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]"
            # sgd with data aug + momentum
            args="${args} -s ${model},data_aug=True,*,lr_schedule=None,momentum=0.9,nesterov=False,weight_decay=0.0]"
            # sgd with data aug + momentum + cosine
            args="${args} -s ${model},data_aug=True,*,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0]"
            # sgd with data aug + momentum + cosine + wd
            args="${args} -s ${model},data_aug=True,*,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]"
            # sgd with data aug + momentum + step + wd
            args="${args} -s ${model},data_aug=True,*,lr_schedule=step,momentum=0.9,nesterov=False,weight_decay=0.0005]"
            # SGD TF
            if [ "${dataset}" != "mnist" ]; then
                model="sgd-tf[batch_size=128"
                # sgd with data aug + momentum + cosine + wd
                args="${args} -s ${model},data_aug=True,*,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]"
            fi
        fi
        $BASIC_CMD $args
    done
done
