#!/bin/bash
#SBATCH --job-name=deq_init_analysis_cifar_reg    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-dev        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=2:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-1

set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"
basic_cmd="python ${vision}/tools/cls_init_analysis.py --n_images 100"

i=0
model_size="LARGE_reg"
lss=("False" "True")  # 2
for ls in ${lss[@]}; do
    opt[$i]="--cfg ${vision}/experiments/cifar/cls_mdeq_${model_size}.yaml --seed 4"
    # when ls is True, add the --ls flag to opt[$i]
    if [ "$ls" == "True" ]; then
        opt[$i]="${opt[$i]} --ls"
    fi
    opt[$i]="${opt[$i]} TRAIN.BEGIN_EPOCH 200"
    i=$((i+1))
done

$basic_cmd ${opt[$SLURM_ARRAY_TASK_ID]}
