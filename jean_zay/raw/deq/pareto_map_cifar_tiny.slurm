#!/bin/bash
#SBATCH --job-name=pareto_cifar_tiny    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:2                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-1600
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"

# seed is going to be the remained of the division of SLURM_ARRAY_TASK_ID by 10
# f_thres is going to be the division of SLURM_ARRAY_TASK_ID by 10 indexing the f_thres_arr
f_thres_arr=($(seq 1 1 40))
b_thres_arr=($(seq 1 1 40))
seed=3
f_thres=${f_thres_arr[$((SLURM_ARRAY_TASK_ID / 40))]}
b_thres=${b_thres_arr[$((SLURM_ARRAY_TASK_ID % 40))]}
output_dir=pareto_map/f${f_thres}_b${b_thres}

# python ${vision}/tools/cls_train.py \
#     --cfg ${vision}/experiments/cifar/cls_mdeq_TINY.yaml \
#     --results_name pareto_map_big.csv \
#     --seed $seed \
#     TRAIN.RESUME False \
#     CUDNN.BENCHMARK False \
#     CUDNN.DETERMINISTIC True \
#     DEQ.F_THRES $f_thres \
#     DEQ.B_THRES $b_thres \
#     DEQ.F_EPS 1e-5 \
#     DEQ.B_EPS 1e-5 \
#     DEQ.RAND_F_THRES_DELTA 0 \
#     OUTPUT_DIR $output_dir

for eval_f_thres in ${f_thres_arr[@]}; do
    python ${vision}/tools/cls_valid.py \
        --cfg ${vision}/experiments/cifar/cls_mdeq_TINY.yaml \
        --results_name pareto_map_big_train_loss.csv \
        --valid_on_train \
        --use_loss_as_perf \
        --seed $seed \
        TEST.MODEL_FILE ${output_dir}/cifar10/cls_mdeq_TINY/final_state_seed$seed.pth.tar \
        DEQ.F_THRES $eval_f_thres \
        DEQ.F_EPS 1e-5
done

extra_f_thres=(45)

for eval_f_thres in ${extra_f_thres[@]}; do
    python ${vision}/tools/cls_valid.py \
        --cfg ${vision}/experiments/cifar/cls_mdeq_TINY.yaml \
        --results_name pareto_map_big_train_loss.csv \
        --valid_on_train \
        --use_loss_as_perf \
        --seed $seed \
        TEST.MODEL_FILE ${output_dir}/cifar10/cls_mdeq_TINY/final_state_seed$seed.pth.tar \
        DEQ.F_THRES $eval_f_thres \
        DEQ.F_EPS 1e-5
done
