#!/bin/bash
#SBATCH --job-name=deq_eq_analysis_imagenet    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=40           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=5:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-31

set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"
basic_cmd="python ${vision}/tools/cls_eq_analysis.py"

i=0
model_size="SMALL_reg"
checkpoints=(50 60 70 80)  # 4
for checkpoint in ${checkpoints[@]}; do
for percent in {0.1,0.3,0.5,1.0}; do  # 4
dropouts=("False" "True")  # 2
for dropout in ${dropouts[@]}; do
    opt[$i]="--cfg ${vision}/experiments/imagenet/cls_mdeq_${model_size}.yaml --percent ${percent}"
    # when dropout is True, add the --dropout_eval flag to opt[$i]
    if [ "$dropout" == "True" ]; then
        opt[$i]="${opt[$i]} --dropout_eval"
    fi
    opt[$i]="${opt[$i]} TRAIN.BEGIN_EPOCH ${checkpoint}"
    i=$((i+1))
done
done
done

$basic_cmd ${opt[$SLURM_ARRAY_TASK_ID]}
