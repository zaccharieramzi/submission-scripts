#!/bin/bash
#SBATCH --job-name=deq_cifar_tiny_overfit    # nom du job
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
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq


python deq/mdeq_vision/tools/cls_train.py \
    --cfg deq/mdeq_vision/experiments/cifar/cls_mdeq_TINY.yaml \
    --seed 42 \
    GPUS 0, \
    TRAIN.WD 0. \
    OUTPUT_DIR overfit_output/ \
    MODEL.DROPOUT 0. \
    MODEL.WNORM false \
    TRAIN.BATCH_SIZE_PER_GPU 128 \
    TRAIN.PRETRAIN_STEPS 0

