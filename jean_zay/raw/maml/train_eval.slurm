#!/bin/bash
#SBATCH --job-name=train_eval    # nom du job
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

. maml.sh

cd $WORK/HowToTrainYourMAMLPytorch

n_steps=$SLURM_ARRAY_TASK_ID

python train_maml_system.py --name_of_args_json_file experiment_config/omniglot_maml-omniglot_1_8_0.1_64_5_1.json\
    --number_of_evaluation_steps_per_iter $n_steps\
    --evaluate_on_test_set_only true\
    --cli_args name_of_args_json_file number_of_evaluation_steps_per_iter evaluate_on_test_set_only
