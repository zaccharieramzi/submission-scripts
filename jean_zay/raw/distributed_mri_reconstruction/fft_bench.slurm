#!/bin/bash
#SBATCH --job-name=fft_profile     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=6           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:2                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-dev        # le calcul va etre long
#SBATCH --partition=gpu_p2
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=2:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. dismri_recon.sh

cd $WORK/flowpm

srun nvprof -f -o fft.nvvp python scripts/fft_benchmark-nvprof.py --cube_size=512 --batch_size=2 --mesh_shape="b1:2" --gpus_per_node=2 --gpus_per_task=2 --layout="nx:b1,tny:b1" > log_1024
