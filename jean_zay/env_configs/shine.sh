#!/bin/bash
module purge
module load python/3.7.5 cuda/10.1.2 cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda
conda activate shine

export FASTMRI_DATA_DIR=$SCRATCH/
export IMAGENET_DIR=$DSDIR/imagenet/RawImages/
export TMP_DIR=$JOBSCRATCH/
export SHINE_LOGS=$SCRATCH/logs
export SHINE_CHECKPOINTS=$SCRATCH/checkpoints
export SHINE_DATA=$SCRATCH/
export SHINE_CONFIG=$WORK/mdeq/experiments/
