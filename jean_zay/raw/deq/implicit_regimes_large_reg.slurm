#!/bin/bash
#SBATCH --job-name=deq_cifar_large_implicit_regimes    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3       # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=2:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-19
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"

seed=$((SLURM_ARRAY_TASK_ID % 10))
checkpoints_arr=(50 200)
checkpoint=${checkpoints_arr[$((SLURM_ARRAY_TASK_ID / 10))]}

python ${vision}/tools/cls_implicit_regime_identification.py \
    --cfg ${vision}/experiments/cifar/cls_mdeq_LARGE_reg.yaml \
    --b_thres_range 1 50 2 \
    --f_thres_range 5 50 5 \
    --n_batches 1 \
    --seed $seed \
    TRAIN.BEGIN_EPOCH $checkpoint \
    TRAIN.WARM_INIT_DIR $JOBSCRATCH \
    DEQ.F_EPS 0.00001 \
    DEQ.B_EPS 0.00001 \
    DEQ.F_SOLVER fixed_point_iteration \
    DEQ.B_SOLVER fixed_point_iteration \
    TRAIN.BATCH_SIZE_PER_GPU 2


