#!/bin/bash
module purge
module load pytorch-gpu/py3/1.7.1

export FASTMRI_DATA_DIR=$SCRATCH/
export IMAGENET_DIR=$SCRATCH/imagenet/
export TMP_DIR=$JOBSCRATCH/
export SHINE_LOGS=$SCRATCH/logs
export SHINE_CHECKPOINTS=$SCRATCH/checkpoints
export SHINE_DATA=$SCRATCH/
export SHINE_CONFIG=$WORK/shine/experiments/
