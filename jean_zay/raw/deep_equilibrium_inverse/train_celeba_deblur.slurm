#!/bin/bash
#SBATCH --job-name=celeba_deblur    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=40           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.err  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deep_equilibrium_inverse.sh

cd $WORK/deep_equilibrium_inverse

python deep_equilibrium_inverse/scripts/fixedpoint/deblur_proxgrad_fixedeta_pre.py \
    --data_path $DSDIR/CelebA/Img/img_align_celeba/ \
    --savepath celeba_equilibriumgrad_blur_save_inf.ckpt \
    --and_maxiters 50
