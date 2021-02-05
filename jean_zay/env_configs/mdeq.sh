#!/bin/bash
module purge
module load python/3.7.5 cuda/10.1.2 cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda
conda activate mdeq

export FASTMRI_DATA_DIR=$SCRATCH/
export TMP_DIR=$JOBSCRATCH/
export MDEQ_LOGS=$SCRATCH/logs
export MDEQ_CHECKPOINTS=$SCRATCH/checkpoints
export MDEQ_DATA=$SCRATCH/
export MDEQ_CONFIG=$WORK/mdeq/experiments/
