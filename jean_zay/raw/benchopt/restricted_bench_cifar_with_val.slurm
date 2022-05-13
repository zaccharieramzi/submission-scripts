#!/bin/bash
#SBATCH --job-name=restricted_benchopt_run
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
#SBATCH --array=0-9

module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH

cd $WORK/benchmark_resnet_classif

i=0

for random_state in {42,43,44,45,46}; do
    BASIC_CMD="benchopt run ."
    BASIC_CMD="$BASIC_CMD -o *18 -d cifar[*,random_state=${random_state},with_validation=True] -r 1 -n 200 --timeout 10800"
    BASIC_CMD="$BASIC_CMD --no-plot -s"

    model="sgd-torch[batch_size=128"
    # sgd with data aug + momentum + cosine + wd
    opt[$i]="$BASIC_CMD ${model},data_aug=True,*,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]"
    i=$((i+1))
    # sgd with data aug + momentum + step + wd
    opt[$i]="$BASIC_CMD ${model},data_aug=True,*,lr_schedule=step,momentum=0.9,nesterov=False,weight_decay=0.0005]"
    i=$((i+1))

done

${opt[$SLURM_ARRAY_TASK_ID]}
