#!/bin/bash
#SBATCH --job-name=unconditional_training    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=40           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-dev        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=2:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. diffusion.sh

cd $WORK/diffusion-function-measures

accelerate launch train_unconditional.py \
  --dataset_name="huggan/pokemon" \
  --resolution=64 \
  --output_dir="ddpm-ema-pokemon-64" \
  --train_batch_size=16 \
  --num_epochs=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no
