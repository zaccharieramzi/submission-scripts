#!/bin/bash
#SBATCH --job-name=benchopt_run_best_sgd
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --qos=qos_gpu-dev
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread

module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH
export TMPDIR=$JOBSCRATCH

cd $WORK/benchmark_resnet_classif

BASIC_CMD="benchopt run ."
BASIC_CMD="$BASIC_CMD -o *18 -d cifar[*,random_state=42,with_validation=False] -r 1 -n 200 --timeout 10800 -f"
ARGS="sgd-tf[batch_size=128,coupled_weight_decay=0.0005,data_aug=True,*,lr_schedule=cosine,momentum=0.9,nesterov=True]"

$BASIC_CMD $ARGS
