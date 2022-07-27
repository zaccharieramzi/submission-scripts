#!/bin/bash
module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH
export TMPDIR=$JOBSCRATCH
export MALLOC_TRIM_THRESHOLD_=0

cd $WORK/benchmark_resnet_classif
