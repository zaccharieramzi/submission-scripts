#!/bin/bash
#SBATCH --job-name=restricted_benchopt_run
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
#SBATCH --array=5

module purge
export PYTHONUSERBASE=$WORK/.local_torch
module load pytorch-gpu/py3/1.10.1
export PATH=$WORK/.local_torch/bin:$PATH
export TMPDIR=$JOBSCRATCH

cd $WORK/benchmark_resnet_classif

i=0

for random_state in {42,}; do
    BASIC_CMD="benchopt run"
    BASIC_CMD="$BASIC_CMD -o '*[model_type=resnet,model_size=18]' -d 'cifar-100[random_state=${random_state},with_validation=False]' -r 1 -n 200 --timeout 10800"
    BASIC_CMD="$BASIC_CMD --no-plot -s"



    # Adam
    model="'adam-torch[batch_size=128"

    # best adam
    opt[$i]="${BASIC_CMD} ${model},coupled_weight_decay=0.0,data_aug=True,decoupled_weight_decay=0.02,lr_schedule=step]'"
    i=$((i+1))

    # SGD Torch
    model="'sgd-torch[batch_size=128"
    # vanilla sgd
    opt[$i]="${BASIC_CMD} ${model},data_aug=False,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]'"
    i=$((i+1))
    # sgd with data aug
    opt[$i]="${BASIC_CMD} ${model},data_aug=True,lr_schedule=None,momentum=0,nesterov=False,weight_decay=0.0]'"
    i=$((i+1))
    # sgd with data aug + momentum
    opt[$i]="${BASIC_CMD} ${model},data_aug=True,lr_schedule=None,momentum=0.9,nesterov=False,weight_decay=0.0]'"
    i=$((i+1))
    # sgd with data aug + momentum + cosine
    opt[$i]="${BASIC_CMD} ${model},data_aug=True,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0]'"
    i=$((i+1))
    # sgd with data aug + momentum + cosine + wd
    opt[$i]="${BASIC_CMD} ${model},data_aug=True,lr_schedule=cosine,momentum=0.9,nesterov=False,weight_decay=0.0005]'"
    i=$((i+1))

    # SGD TF
    model="sgd-tf[batch_size=128"
    # sgd with data aug + momentum + cosine + wd
    opt[$i]="${BASIC_CMD} ${model},coupled_weight_decay=0.0005,data_aug=True,decoupled_weight_decay=0.0lr_schedule=cosine,momentum=0.9,nesterov=True]'"
    i=$((i+1))
done

${opt[$SLURM_ARRAY_TASK_ID]}
