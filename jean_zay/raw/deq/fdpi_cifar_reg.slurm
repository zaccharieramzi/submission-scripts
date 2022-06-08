#!/bin/bash
#SBATCH --job-name=fdpi_cifar_reg    # nom du job
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
#SBATCH --array=50-199

set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"
basic_cmd="python ${vision}/tools/cls_fixed_point_da_invariance.py"

i=0
model_size="LARGE_reg"
if [ "$model_size" == "TINY" ]; then
    checkpoints=(14 40)  # 4
else
    checkpoints=(10 50 100 150 200)  # 5
fi
for checkpoint in ${checkpoints[@]}; do
for seed in {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}; do  # 20
if [ "$model_size" == "TINY" ]; then
    dropouts=("False")  # 1
else
    dropouts=("False" "True")  # 2
fi
for dropout in ${dropouts[@]}; do
    opt[$i]="--cfg ${vision}/experiments/cifar/cls_mdeq_${model_size}.yaml --seed ${seed}"
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
