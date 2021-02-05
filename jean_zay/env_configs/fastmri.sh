#!/bin/bash
module purge
module load tensorflow-gpu/py3/2.2.0

export FASTMRI_DATA_DIR=$SCRATCH/
export OASIS_DATA_DIR=$SCRATCH/OASIS_data
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
export TMP_DIR=$JOBSCRATCH/
export SINGLECOIL_TRAIN_DIR=singlecoil_train/singlecoil_train
