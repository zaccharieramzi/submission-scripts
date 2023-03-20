#!/bin/bash
#SBATCH --job-name=evaluation    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=2:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=1-20

cd $WORK/submission-scripts/jean_zay/env_configs/

. imaml.sh

cd $WORK/imaml_dev/examples

n_steps=$(($SLURM_ARRAY_TASK_ID * 2))

python measure_accuracy.py --load_agent 5_way_1_shot_exp1/final_model.pickle --N_way 5 --K_shot 1 --num_tasks 600 --n_steps $n_steps --lam 0.0 --inner_lr 1e-1 --task Omniglot --data_dir $SCRATCH/omniglot-py
