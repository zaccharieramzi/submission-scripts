#!/bin/bash
#SBATCH --job-name=learnlet_undecimated_exact_rec     # nom du job
#SBATCH --ntasks=4                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=4          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=10:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=learnlet_undecimated_exact_rec%j.out # nom du fichier de sortie
#SBATCH --error=learnlet_undecimated_exact_rec%j.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/understanding-unets

. ./submission_scripts_jean_zay/env_config.sh

srun --ntasks=1 python ./learning_wavelets/training_scripts/learnlet_subclassed_training.py -e -nf 64 -u -gpus 0,1&
srun --ntasks=1 python ./learning_wavelets/training_scripts/learnlet_subclassed_training.py -nf 64 -u -gpus 2,3&

wait  # wait for all ccc_mprun(s) to complete.
