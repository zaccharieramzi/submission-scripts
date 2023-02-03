#!/bin/bash
#SBATCH --job-name=pareto_cityscapes    # nom du job
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
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=2-15
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"

f_thres=$($SLURM_ARRAY_TASK_ID * 3)

python ${vision}/tools/seg_test.py \
    --cfg ${vision}/experiments/cityscapes/seg_mdeq_XL.yaml \
    --results_name cityscapes_n_iter_results.csv \
    TEST.MODEL_FILE ${vision}/MDEQ_XL_Seg.pkl \
    DEQ.F_EPS 0.0000001 \
    DEQ.F_THRES $f_thres
