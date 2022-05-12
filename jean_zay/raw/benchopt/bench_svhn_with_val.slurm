#!/bin/bash
#SBATCH --job-name=benchopt_run
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --qos=qos_gpu-t3
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread
#SBATCH --array=0-95

module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH

cd $WORK/benchmark_resnet_classif

BASIC_CMD="benchopt run ."
BASIC_CMD="$BASIC_CMD -o *18 -d svhn[*,random_state=42,with_validation=True] -r 1 -n 200 --timeout 10800"
BASIC_CMD="$BASIC_CMD --no-plot -s"



# Adam
i=0
for model in {'tf','torch'}; do
    framework="${model}"
    model="adam-${model}[batch_size=128"
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
    framework="${model}"
    model="sgd-${model}[batch_size=128"
    for data_aug in {'False','True'}; do
        for momentum in {'0','0.9'}; do
            for nesterov in {'False','True'}; do
                if [ "$nesterov" == "True" ] && [ "$momentum" = "0" ]; then
                    continue
                fi
                for wd in {'0.0','0.0005'}; do
                    for lr in {'None','step','cosine'}; do
                        opt[$i]="${model},data_aug=${data_aug},*,lr_schedule=${lr},momentum=${momentum},nesterov=${nesterov},weight_decay=${wd}]"
                        i=$((i+1))
                    done
                done
            done
        done
    done
done

$BASIC_CMD ${opt[$SLURM_ARRAY_TASK_ID]}
