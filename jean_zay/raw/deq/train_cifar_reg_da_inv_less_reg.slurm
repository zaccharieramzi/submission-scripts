#!/bin/bash
#SBATCH --job-name=deq_cifar_reg_da_inv_less_reg    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t4      # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=100:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"

python ${vision}/tools/cls_train.py \
    --cfg ${vision}/experiments/cifar/cls_mdeq_LARGE_reg_da_inv.yaml \
    --save_at 10 50 100 150 200 \
    --seed $SLURM_ARRAY_TASK_ID \
    TRAIN.RESUME False \
    CUDNN.BENCHMARK False \
    CUDNN.DETERMINISTIC True \
    OUTPUT_DIR da_inv_xp_output \
    LOSS.JAC_LOSS_WEIGHT 0.2

