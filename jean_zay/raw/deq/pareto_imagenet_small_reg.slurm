#!/bin/bash
#SBATCH --job-name=pareto_imagenet_small_reg    # nom du job
#SBATCH --ntasks=1                # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=40           # nombre de coeurs à réserver par tâche
#SBATCH --gres=gpu:4                 # nombre de GPU à réserver par nœud
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-t4        # le calcul va etre long
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=100:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out # nom du fichier de sortie
#SBATCH --error=%x_%A_%a.err  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-5
set -x
cd $WORK/submission-scripts/jean_zay/env_configs/

. deq.sh

cd $WORK/deq

vision="deq/mdeq_vision"

f_thres_arr=(1 2 5 10 12 14)
f_thres=${f_thres_arr[$((SLURM_ARRAY_TASK_ID))]}
seed=0
output_dir=pareto_output_f$f_thres

python ${vision}/tools/cls_train.py \
    --cfg ${vision}/experiments/imagenet/cls_mdeq_SMALL_reg.yaml \
    --results_name imagenet_results.csv \
    --seed $seed \
    DEQ.F_THRES $f_thres \
    OUTPUT_DIR $output_dir

for eval_f_thres in ${f_thres_arr[@]}; do
    python ${vision}/tools/cls_valid.py \
        --cfg ${vision}/experiments/imagenet/cls_mdeq_SMALL_reg.yaml \
        --results_name imagenet_results.csv \
        --seed $seed \
        TEST.MODEL_FILE ${output_dir}/imagenet/cls_mdeq_SMALL_reg/final_state_seed$seed.pth.tar \
        DEQ.F_THRES $eval_f_thres
done

extra_f_thres=(16 20)

for eval_f_thres in ${extra_f_thres[@]}; do
    python ${vision}/tools/cls_valid.py \
        --cfg ${vision}/experiments/imagenet/cls_mdeq_SMALL_reg.yaml \
        --results_name imagenet_results.csv \
        --seed $seed \
        TEST.MODEL_FILE ${output_dir}/imagenet/cls_mdeq_SMALL_reg/final_state_seed$seed.pth.tar \
        DEQ.F_THRES $eval_f_thres
done
