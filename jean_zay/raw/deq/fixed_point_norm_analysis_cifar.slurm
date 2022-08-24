#!/bin/bash
#SBATCH --job-name=fixed_point_norm_cifar_reg    # nom du job
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
basic_cmd="python ${vision}/tools/cls_eq_analysis.py"
output_dir=pareto_output_f8

i=0
model_sizes=("LARGE_reg" "LARGE_reg_warm")  # 4
for model_size in ${model_sizes[@]}; do
    opt[$i]="--cfg ${vision}/experiments/cifar/cls_mdeq_${model_size}.yaml --seed 4"
    opt[$i]="${opt[$i]} TRAIN.BEGIN_EPOCH ${checkpoint} DEQ.F_THRES 40 DEQ.F_EPS 0.00001"
    opt[$i]="${opt[$i]} TEST.MODEL_FILE ${output_dir}/cifar10/cls_mdeq_LARGE_reg/final_state_seed4.pth.tar"
    i=$((i+1))
done

$basic_cmd ${opt[$SLURM_ARRAY_TASK_ID]}
