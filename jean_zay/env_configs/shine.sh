#!/bin/bash
module purge
module load pytorch-gpu/py3/1.6.0

export FASTMRI_DATA_DIR=$SCRATCH/
export IMAGENET_DIR=$SCRATCH/imagenet/
export CIFAR_DIR=$SCRATCH/cifar/
export TMP_DIR=$JOBSCRATCH/
export SHINE_LOGS=$SCRATCH/logs
export SHINE_CHECKPOINTS=$SCRATCH/checkpoints
export SHINE_DATA=$SCRATCH/
export SHINE_CONFIG=$WORK/shine/experiments/
export WORK_DIR=$WORK/shine/
