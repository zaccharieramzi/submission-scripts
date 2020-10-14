#!/bin/bash
#MSUB -r train_unets_different_n_filters               # Request name
#MSUB -n 4                         # Number of tasks to use
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

ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py -bnf 32 -gpus 0 &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py -bnf 16 -gpus 1 &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py -bnf 8 -gpus 2 &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/unet_training.py -bnf 4 -gpus 3 &

wait  # wait for all ccc_mprun(s) to complete.
