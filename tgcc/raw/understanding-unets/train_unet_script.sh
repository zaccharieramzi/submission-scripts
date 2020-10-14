#!/bin/bash
#MSUB -r train_unets                # Request name
#MSUB -n 2                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -x
#MSUB -T 86400                      # Elapsed time limit in seconds
#MSUB -o unet_train_%I.o              # Standard output. %I is the job id
#MSUB -e unet_train_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -Q normal
#MSUB -m scratch,work
#MSUB -@ zaccharie.ramzi@gmail.com:begin,end
#MSUB -A gch0424                  # Project ID

set -x
cd $workspace/understanding-unets

. ./submission_scripts/env_config.sh

ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py --ns-train 20 40 -gpus 01 &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py --ns-train 30 30 -gpus 23 &

wait  # wait for all ccc_mprun(s) to complete.
