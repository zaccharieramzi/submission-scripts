#!/bin/bash
#SBATCH --job-name=transformer     # nom du job
#SBATCH --ntasks=4                # nombre de tâche MPI
#SBATCH --ntasks-per-node=4          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t4        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=100:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. shine.sh

cd $WORK/deq-shine/DEQModel

./run_wt103_deq_transformer.sh train\
  --data $WIKITEXT_DIR\
  --work_dir $SHINE_CHECKPOINTS\
  --b_thres 35\
  --restart \
  --restart_dir $SHINE_CHECKPOINTS/deq-wt103/20210413-150921 \
  --start_train_steps 128400\
  --pretrain_steps 0
