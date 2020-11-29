#!/bin/bash

# Parameters
#SBATCH --account=hih@gpu
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=40
#SBATCH --distribution=block:block
#SBATCH --error=%A_%a_0_log.err
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=%A_%a_0_log.out
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t4
#SBATCH --signal=USR1@90
#SBATCH --time=100:00:00
#SBATCH --wckey=submitit


echo $SLURM_ARRAY_TASK_ID
