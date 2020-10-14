#!/bin/bash
#MSUB -r test_multi_gpu                # Request name
#MSUB -n 4                         # Number of tasks to use
#MSUB -c 2                         # I want 2 cores per task since io might be costly
#MSUB -T 1800                      # Elapsed time limit in seconds
#MSUB -o test_%I.o              # Standard output. %I is the job id
#MSUB -e test_%I.e              # Error output. %I is the job id
#MSUB -q v100               # Queue
#MSUB -Q test  # this is just a test script
#MSUB -m scratch,work
#MSUB -@ zaccharie.ramzi@gmail.com:begin,end
#MSUB -A gch0424                  # Project ID

set -x
module load cuda/10.1.105
module load python3/3.7.5
cd $workspace/understanding-unets

ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/test.py  &
ccc_mprun -E '--exclusive' -n 1 python3 ./learning_wavelets/training_scripts/test.py  &

wait  # wait for all ccc_mprun(s) to complete.
