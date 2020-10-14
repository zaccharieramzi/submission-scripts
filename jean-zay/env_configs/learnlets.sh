#!/bin/bash
module purge
module load tensorflow-gpu/py3/2.2.0-dev

export BSD500_DATA_DIR=$SCRATCH/
export BSD68_DATA_DIR=$SCRATCH/
export DIV2K_DATA_DIR=$SCRATCH/
export LOGS_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/
