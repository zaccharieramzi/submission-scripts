#!/bin/bash
module purge
module load tensorflow-gpu/py3/2.7.0

export CHKPT_PATH=$SCRATCH/checkpoints/ifip/
export LOG_PATH=$SCRATCH/logs/ifip/
export KERAS_HOME=$SCRATCH/.keras/
export WANDB_PATH=$SCRATCH/wandb/ifip/
export WANDB_MODE=offline