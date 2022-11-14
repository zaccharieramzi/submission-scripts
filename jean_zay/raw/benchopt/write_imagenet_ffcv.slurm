#!/bin/bash
#SBATCH --job-name=write_ffcv_imagenet
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --partition=archive
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --qos=qos_gpu-dev

set -x

# Required environmental variables for the script:
export IMAGENET_DIR=$SCRATCH/RawImages
export WRITE_DIR=$SCRATCH/FFCVImageNet
mkdir -p $WRITE_DIR

cd $WORK/submission-scripts/jean_zay/env_configs;
. benchopt.sh
module load libjpeg-turbo/2.1.3

# Starting in the root of the Git repo:
cd $WORK/ffcv-imagenet;

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90
