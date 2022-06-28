#!/bin/bash
#SBATCH --job-name=pareto_cifar_reg_warm    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-59
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"

# seed is going to be the remained of the division of SLURM_ARRAY_TASK_ID by 10
# f_thres is going to be the division of SLURM_ARRAY_TASK_ID by 10 indexing the f_thres_arr
f_thres_arr=(1 2 4 6 8 10)
seed=$((SLURM_ARRAY_TASK_ID % 10))
f_thres=f_thres=${f_thres_arr[$((SLURM_ARRAY_TASK_ID / 10))-1]}

python ${vision}/tools/cls_train.py \
    --cfg ${vision}/experiments/cifar/cls_mdeq_LARGE_reg_warm.yaml \
    --results_name results.csv \
    --seed $seed \
    TRAIN.RESUME False \
    CUDNN.BENCHMARK False \
    CUDNN.DETERMINISTIC True \
    DEQ.F_THRES $f_thres \
    OUTPUT_DIR pareto_output
