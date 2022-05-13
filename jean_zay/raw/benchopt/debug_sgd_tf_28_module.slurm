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
export PYTHONUSERBASE=$WORK/.local_tf2.8
module load tensorflow-gpu/py3/2.8.0
export PATH=$WORK/.local_tf2.8/bin:$PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/gpfslocalsys/cuda/11.2"

cd $WORK/benchmark_resnet_classif

BASIC_CMD="benchopt run ."
BASIC_CMD="$BASIC_CMD -o *18 -d cifar[framework=tensorflow,random_state=42,with_validation=False] -r 1 -n 1 --timeout 3600 -f"
ARGS="sgd-tf[batch_size=128,coupled_weight_decay=0.0,data_aug=False,decoupled_weight_decay=0.0,*,lr_schedule=None,momentum=0.9,nesterov=False]"

$BASIC_CMD $ARGS
