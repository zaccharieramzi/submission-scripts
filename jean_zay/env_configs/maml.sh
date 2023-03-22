#!/bin/bash
module purge
module load pytorch-gpu/py3/1.11.0


export DATASET_DIR="datasets"
export CUDA_VISIBLE_DEVICES=0
