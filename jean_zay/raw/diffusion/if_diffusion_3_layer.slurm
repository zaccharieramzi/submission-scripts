#!/bin/bash
#SBATCH --job-name=if_diffusion_3_layer   # nom du job
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
#SBATCH --array=0
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. diffusion.sh

cd $WORK/diffusion-function-measures

accelerate launch diffusion_function_measures/training/main_if.py\
    --config-name=main_if_diffusion_3_layer \
    data.dataloader.num_workers=0 \
    +data.image_dataset.sample=1 \
    data.if_dataset.if_representations_path="${SCRATCH}/if_representations_debug_grayscale_3_layer" \
    data.image_dataset.dataset_path="${SCRATCH}/pokemon" \
    training.n_epochs=10000 \
    training.log_every=100 \
    training.save_every=1000
