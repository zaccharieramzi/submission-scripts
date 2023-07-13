#!/bin/bash
#SBATCH --job-name=pareto_ldeq_video    # nom du job
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
#SBATCH --array=1-25
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. ldeq_rwr.sh

cd $WORK/LDEQ_RwR


f_thres=$SLURM_ARRAY_TASK_ID

python test_LDEQ_WFLW_V.py --landmark_model_weights final.pth.tar --WFLW_V_dataset_path $SCRATCH/WFLW_V_release --WFLW_V_workers 8 --WFLW_V_batch_size 5 \
    --n-forward $f_thres --output-csv pareto_ldeq_video.csv --WFLW_V_split easy
