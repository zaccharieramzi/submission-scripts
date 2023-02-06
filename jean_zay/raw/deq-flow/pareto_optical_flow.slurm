#!/bin/bash
#SBATCH --job-name=pareto_optical_flow    # nom du job
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
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-10
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq_flow.sh

cd $WORK/deq-flow


f_thres=$(($SLURM_ARRAY_TASK_ID * 15))

python deq_flow/main.py --eval --name deq-flow-H-all-grad --stage things \
    --validation kitti sintel --restore_ckpt checkpoints/deq-flow-H-things-test-3x.pth --gpus 0 \
    --wnorm --eval_f_thres $f_thres --f_solver naive_solver  \
    --huge --results_name optical_flow_results.csv
