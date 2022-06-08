#!/bin/bash
module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH
export TMPDIR=$JOBSCRATCH

cd $WORK/benchmark_resnet_classif
