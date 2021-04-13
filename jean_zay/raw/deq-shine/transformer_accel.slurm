#!/bin/bash
#SBATCH --job-name=transformer_accel     # nom du job
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
#SBATCH --array=0-1
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. shine.sh

cd $WORK/deq-shine/DEQModel

opt[0]="--shine --name shine"
opt[1]="--fpn --name fpn"

./run_wt103_deq_transformer.sh train\
  --data $WIKITEXT_DIR \
  --work_dir $SHINE_CHECKPOINTS\
  # --restart_dir $SHINE_CHECKPOINTS/deq\
  ${opt[$SLURM_ARRAY_TASK_ID]}
