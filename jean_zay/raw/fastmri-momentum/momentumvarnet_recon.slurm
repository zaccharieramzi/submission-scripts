#!/bin/bash
#SBATCH --job-name=momentum_varnet     # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=24           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:8                # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --partition=gpu_p2

set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. momentum.sh

cd $WORK/fastmri-momentum/

opt[0]='--use_momentum --gamma 0.2'
# opt[1]='--use_momentum'
# opt[2]='--use_momentum --use_backprop'


python fastmri_momentum/knee_training.py \
    --num_cascades 6 \
    --pools 3 \
    --chans 8 \
    --sens_pools 2 \
    --data_path $ROOT_DIR \
    --gpus 8 \
    --max_epochs 20 \
    --default_root_dir $ROOT_DIR \
    ${opt[0]}