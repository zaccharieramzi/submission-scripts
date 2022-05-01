#!/bin/bash
#SBATCH --job-name=benchopt_run
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --qos=qos_gpu-t3
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread
#SBATCH --array=0-48

. $WORK/submission-scripts/jean_zay/env_configs/benchopt.sh
cd $WORK/benchmark_resnet_classif

BASIC_CMD="benchopt run ."
BASIC_CMD="$BASIC_CMD -o *18 -d cifar -r 1 -n 200 --timeout 7200 -s"



# Adam
i=0
for model in {'tf','torch'}; do
    model="adam-${model}[batch_size=64"
    for data_aug in {'False','True'}; do
        for wd in {'0.0','0.02'}; do
            for lr in {'None','step','cosine'}; do
                opt[$i]="${model},coupled_weight_decay=0.0,data_aug=${data_aug},decoupled_weight_decay=${wd},*,lr_schedule=${lr}]"
                i=$((i+1))
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
                    opt[$i]="${model},data_aug=${data_aug},*,lr_schedule=${lr},*,nesterov=${nesterov},weight_decay=${wd}]"
                    i=$((i+1))
                done
            done
        done
    done
done

$BASIC_CMD ${opt[$SLURM_ARRAY_TASK_ID]}
