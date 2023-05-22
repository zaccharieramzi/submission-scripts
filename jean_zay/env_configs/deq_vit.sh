#!/bin/bash
module purge
module load python/3.10.4
module load cuda/11.7.1
module load cudnn/8.5.0.96-11.7-cuda

cd $xpa_ALL_CCFRWORK/deq-vit
