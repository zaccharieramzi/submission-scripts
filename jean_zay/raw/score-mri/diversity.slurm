#!/bin/bash
#SBATCH --job-name=diversity    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-dev        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=02:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-3
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. score_mri.sh

cd $WORK/score-MRI

accelerations=(4 8 16 32)
center_fractions=(0.08 0.04 0.02 0.01)

acc=${accelerations[$SLURM_ARRAY_TASK_ID]}
cf=${center_fractions[$SLURM_ARRAY_TASK_ID]}

python score_mri/diversity_single_coil.py \
    --data 001 \
    --N 500 \
    --acc_factor $acc \
    --center_fraction $cf \
    --save_dir "results/diversity_${acc}_${cf}/" \
    --n_images 10
