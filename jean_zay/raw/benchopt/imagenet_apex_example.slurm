#!/bin/bash
#SBATCH --job-name=imagenet_apex_example
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --qos=qos_gpu-dev
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread

cd $WORK/submission-scripts/jean_zay/env_configs

. benchopt_2.sh

cd $WORK/submission-scripts/jean_zay/raw/benchopt

python imagenet_apex.py \
    $SCRATCH/RawImages \
    -j 10 \
    -b 128 \
    -p 1 \
    --opt-level O0
