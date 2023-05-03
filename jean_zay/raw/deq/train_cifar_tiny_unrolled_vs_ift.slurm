#!/bin/bash
#SBATCH --job-name=stability_unrolled_vs_ift    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:2                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t3       # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-19
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"

f_thres=9
seed=$((SLURM_ARRAY_TASK_ID % 10))
# unrolled is a binary worth "True" if SLURM_ARRAY_TASK_ID is greater than 9 and "False" otherwise
unrolled=$((SLURM_ARRAY_TASK_ID > 9))
# dir is ift if unrolled is False and unrolled if unrolled is True
dir=$(if [ $unrolled = 1 ]; then echo "unrolled"; else echo "ift"; fi)
output_dir=stability/${dir}_${f_thres}
unrolled=$(if [ $unrolled = 1 ]; then echo "True"; else echo "False"; fi)

python ${vision}/tools/cls_train.py \
    --cfg ${vision}/experiments/cifar/cls_mdeq_TINY.yaml \
    --seed $seed \
    TRAIN.ALL_UNROLLED $unrolled \
    DEQ.F_THRES $f_thres \
    CUDNN.BENCHMARK False \
    CUDNN.DETERMINISTIC True \
    OUTPUT_DIR $output_dir

f_thres_eval=(3 5 6 9 10 11 15 16 18 20 25 30 35)
for eval_f_thres in ${f_thres_eval[@]}; do
    python ${vision}/tools/cls_valid.py \
        --cfg ${vision}/experiments/cifar/cls_mdeq_TINY.yaml \
        --results_name stability_results.csv \
        --seed $seed \
        TEST.MODEL_FILE ${output_dir}/cifar10/cls_mdeq_TINY/final_state_seed$seed.pth.tar \
        DEQ.F_THRES $eval_f_thres
done
