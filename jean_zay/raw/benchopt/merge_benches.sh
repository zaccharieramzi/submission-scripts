#!/bin/bash
DATASET="$1"
WITH_VALIDATION="$2"

module purge

cd $WORK/benchmark_resnet_classif

BASIC_CMD="benchopt run ."
BASIC_CMD="$BASIC_CMD -o *18 -d $DATASET[*,random_state=42,with_validation=$WITH_VALIDATION] -r 1 -n 200 --timeout 10800"

args=''

# Adam
i=0
for model in {'tf','torch'}; do
    framework="${model}"
    model="adam-${model}[batch_size=128"
    for data_aug in {'False','True'}; do
        for wd in {'0.0','0.02'}; do
            for lr in {'None','step','cosine'}; do
                args="${args} -s ${model},coupled_weight_decay=0.0,data_aug=${data_aug},decoupled_weight_decay=${wd},*,lr_schedule=${lr}]"
            done
        done
    done
done


# SGD
for model in {'tf','torch'}; do
    framework="${model}"
    model="sgd-${model}[batch_size=128"
    for data_aug in {'False','True'}; do
        for nesterov in {'False','True'}; do
            for wd in {'0.0','0.0005'}; do
                for lr in {'None','step','cosine'}; do
                    args="${args} -s ${model},data_aug=${data_aug},*,lr_schedule=${lr},momentum=${momentum},nesterov=${nesterov},weight_decay=${wd}]"
                done
            done
        done
    done
done
export PYTHONUSERBASE=$WORK/.local_tf
 module load tensorflow-gpu/py3/2.7.0
export PATH=$WORK/.local_tf/bin:$PATH
$BASIC_CMD $args
