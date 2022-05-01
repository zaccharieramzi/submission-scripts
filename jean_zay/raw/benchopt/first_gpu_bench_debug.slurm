#!/bin/bash
#SBATCH --job-name=benchopt_run
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#SBATCH --qos=qos_gpu-dev
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread
#SBATCH --array=0-0

module purge

cd $WORK/benchmark_resnet_classif

BASIC_CMD="benchopt run ."
BASIC_CMD="$BASIC_CMD -o *18 -d cifar -r 1 -n 2 --timeout 200 -s"



# Adam
i=0
for model in {'tf','torch'}; do
    framework="${model}"
    model="adam-${model}[batch_size=64"
    for data_aug in {'False','True'}; do
        for wd in {'0.0','0.02'}; do
            for lr in {'None','step','cosine'}; do
                opt[$i]="${model},coupled_weight_decay=0.0,data_aug=${data_aug},decoupled_weight_decay=${wd},*,lr_schedule=${lr}]"
                # activate tensorflow if i is the slurm array task id and the framework is tf
                # otherwise activate torch
                if [ "$i" == "$SLURM_ARRAY_TASK_ID" ]; then
                    if [ "$framework" == "tf" ]; then
                        echo "activating tensorflow"
                        export PYTHONUSERBASE=$WORK/.local_tf
                        module load tensorflow-gpu/py3/2.7.0
                        export PATH=$WORK/.local_tf/bin:$PATH
                    else
                        echo "activating torch"
                        export PYTHONUSERBASE=$WORK/.local_torch
                        module load pytorch-gpu/py3/1.10.1
                        export PATH=$WORK/.local_torch/bin:$PATH
                    fi
                fi
                i=$((i+1))
            done
        done
    done
done


# SGD
for model in {'tf','torch'}; do
    framework="${model}"
    model="sgd-${model}[batch_size=64"
    for data_aug in {'False','True'}; do
        for nesterov in {'False','True'}; do
            for wd in {'0.0','0.0001'}; do
                for lr in {'None','step','cosine'}; do
                    opt[$i]="${model},data_aug=${data_aug},*,lr_schedule=${lr},*,nesterov=${nesterov},weight_decay=${wd}]"
                    if [ "$i" == "$SLURM_ARRAY_TASK_ID" ]; then
                        if [ "$framework" == "tf" ]; then
                            echo "activating tensorflow"
                            export PYTHONUSERBASE=$WORK/.local_tf
                            module load tensorflow-gpu/py3/2.7.0
                            export PATH=$WORK/.local_tf/bin:$PATH
                        else
                            echo "activating torch"
                            export PYTHONUSERBASE=$WORK/.local_torch
                            module load pytorch-gpu/py3/1.10.1
                            export PATH=$WORK/.local_torch/bin:$PATH
                        fi
                    fi
                    i=$((i+1))
                done
            done
        done
    done
done

$BASIC_CMD ${opt[0]}
