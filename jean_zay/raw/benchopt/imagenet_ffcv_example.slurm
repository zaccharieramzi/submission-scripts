#!/bin/bash
#SBATCH --job-name=imagenet_ffcv_example
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

. benchopt.sh
module load libjpeg-turbo/2.1.3

cd $WORK/submission-scripts/jean_zay/raw/benchopt


python imagenet_ffcv.py \
    --data.train_dataset=$SCRATCH/FFCVImageNet/train_500_0.50_90.ffcv \
    --data.val_dataset=$SCRATCH/FFCVImageNet/val_500_0.50_90.ffcv \
    --data.num_workers=10 --data.in_memory=1 \
    --logging.folder=$WORK/benchopt/ffcv_logs/ \
    --training.batch_size=128 --training.epochs=1 \
    --training.label_smoothing=0. \
    --resolution.min_res=224 --resolution.max_res=224
