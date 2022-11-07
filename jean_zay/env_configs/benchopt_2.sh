#!/bin/bash
module purge
export PYTHONUSERBASE=$WORK/.local_torch_1_12
module load pytorch-gpu/py3/1.12.1
export PATH=$WORK/.local_torch_1_12/bin:$PATH
export TMPDIR=$JOBSCRATCH
export MALLOC_TRIM_THRESHOLD_=0

cd $WORK/benchmark_resnet_classif
