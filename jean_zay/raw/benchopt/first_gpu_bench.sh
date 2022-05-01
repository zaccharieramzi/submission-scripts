#!/bin/bash
SBATCH="sbatch -A xpa@v100 --time 03:00:00 --gres gpu:1 --cpu-pre-task 10 --qos=qos_gpu-t3"
BASIC_CMD="$SBATCH benchopt run benchmark_resnet_classif"
BASIC_CMD="$BASIC_CMD -o *18 -d cifar -r 1 -n 200"

# Adam
for model in {'tf','torch'}; do
    model="adam-${model}[batch_size=64"
    for data_aug in {'False','True'}; do
        for wd in {'0.0','0.02'}; do
            for lr in {'None','step','cosine'}; do
                $BASIC_CMD --timeout 7200 -s "${model},coupled_weight_decay=0.0,data_aug=${data_aug},decoupled_weight_decay=${wd},lr_schedule=${lr}]"
            done
        done
    done
done


# SGD
for model in {'tf','torch'}; do
    model="sgd-${model}[batch_size=64"
    for data_aug in {'False','True'}; do
        for nesterov in {'False','True'}; do
            for wd in {'0.0','0.0001'}; do
                for lr in {'None','step','cosine'}; do
                    $BASIC_CMD --timeout 7200 -s "${model},data_aug=${data_aug},*,lr_schedule=${lr},nesterov=${nesterov},weight_decay=${wd}]"
                done
            done
        done
    done
done
