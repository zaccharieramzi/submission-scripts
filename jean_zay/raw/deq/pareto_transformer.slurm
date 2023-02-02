#!/bin/bash
#SBATCH --job-name=pareto_transformer     # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=40           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=2:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
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
  --f_eps 0.0000001 \
  --f_solver broyden \
  --pretrain_step 0 \
  --debug \
  --results_file transformer_results.csv
