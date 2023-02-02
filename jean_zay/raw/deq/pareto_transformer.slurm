#!/bin/bash
#SBATCH --job-name=pareto_transformer     # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-20
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh
sequence="deq/deq_sequence"


cd $WORK/deq/$sequence


F_THRES=$(($SLURM_ARRAY_TASK_ID * 3))

. wt103_deq_transformer.sh train \
  --data $WIKITEXT_DIR \
  --load pretrained_wt103_deqtrans_v3.pkl \
  --name eval_$SLURM_ARRAY_TASK_ID \
  --eval \
  --mem_len 300 \
  --f_thres $F_THRES \
  --pretrain_step 0 \
  --results_file transformer_results.csv
