#!/bin/bash
#SBATCH --job-name=pareto_optical_flow_w_options    # nom du job
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
#SBATCH --array=0-29
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq_flow.sh

cd $WORK/deq-flow


f_thres=$((($SLURM_ARRAY_TASK_ID+1) % 10 * 9))
options=$(($SLURM_ARRAY_TASK_ID % 10))
# if options is 0, then call the script with both the --fixed_point and the --warm_start options
# if options is 1, then call the script with only the --fixed_point option
# if options is 2, then call the script with only the --warm_start option

script="python deq_flow/main.py --eval --name deq-flow-H-all-grad --stage things \
    --validation sintel --restore_ckpt checkpoints/deq-flow-H-things-test-3x.pth --gpus 0 \
    --wnorm --eval_f_thres $f_thres --f_eps 0.00001 --f_solver broyden  \
    --huge --results_name optical_flow_results_w_options.csv"

if [ $options -eq 0 ]
then
    script="$script --fixed_point --warm_start"
elif [ $options -eq 1 ]
then
    script="$script --fixed_point"
elif [ $options -eq 2 ]
then
    script="$script --warm_start"
fi

