#!/bin/bash
#SBATCH --job-name=single_coil_test   # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t4        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=60:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-1
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. score_mri.sh

cd $WORK/score-MRI

accelerations=(4 8)

python score_mri/inference_single_coil_fastmri.py \
    --data $SCRATCH \
    --N 500 \
    --batch_size 1 \
    --mode test \
    --acceleration ${accelerations[$SLURM_ARRAY_TASK_ID]}
