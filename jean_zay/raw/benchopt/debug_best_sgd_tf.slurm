#!/bin/bash
#SBATCH --job-name=benchopt_run_debug
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --qos=qos_gpu-dev
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread

module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/11.2"

cd $WORK/benchmark_resnet_classif

BASIC_CMD="benchopt run ."
BASIC_CMD="$BASIC_CMD -o *18 -d cifar[*,random_state=42,with_validation=False] -r 1 -n 1 --timeout 3600 -f"
ARGS="sgd-tf[batch_size=128,data_aug=True,*,lr_schedule=cosine,*,nesterov=True,weight_decay=0.0005]"

$BASIC_CMD $ARGS
