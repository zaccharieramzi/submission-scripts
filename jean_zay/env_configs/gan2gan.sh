#!/bin/bash
module purge
module load tensorflow-gpu/py3/2.5.0

export GAN2GAN_LOGS=$SCRATCH
export GAN2GAN_CKPT=$SCRATCH
export BSD_DATA=$SCRATCH