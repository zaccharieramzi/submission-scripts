#!/bin/bash
#SBATCH --job-name=training    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t4        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=30:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)

cd $WORK/submission-scripts/jean_zay/env_configs/

. imaml.sh

cd $WORK/imaml_dev/examples

python omniglot_implicit_maml.py --save_dir 5_way_1_shot_exp1 --N_way 5 --K_shot 1 --inner_lr 1e-1 --outer_lr 1e-3 --n_steps 16 --meta_steps 20000 --num_tasks 200000 --task_mb_size 32 --lam 2.0 --cg_steps 5 --cg_damping 1.0 --load_tasks ./task_defs/Omniglot_5_way_1_shot.pickle --data_dir $SCRATCH/omniglot-py
