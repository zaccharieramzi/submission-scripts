#!/bin/bash
module purge
module load python/3.7.5 cuda/10.1.2 cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda
conda activate soft-thresholding

export BSD_DATA_DIR=$SCRATCH/
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
