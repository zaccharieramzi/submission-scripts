#!/bin/bash
#SBATCH --job-name=mcncpdnet_eval     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=40           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=60:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=mcncpdnet_eval%A_%a.out # nom du fichier de sortie
#SBATCH --error=mcncpdnet_eval%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-1

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

export RUN_ID_RADIAL="ncpdnet_sense___rfs_radial_compound_mssim_dcomp_1596636372"
export RUN_ID_SPIRAL="ncpdnet_sense___rfs_spiral_compound_mssim_dcomp_1596636369"

opt[0]="-a 4 -m pdnet -rfs -mc -t radial -e 70 -n 50 -r ${RUN_ID_RADIAL}"
opt[1]="-a 4 -m pdnet -rfs -mc -t spiral -e 70 -n 50 -r ${RUN_ID_SPIRAL}"

srun python ./fastmri_recon/evaluate/scripts/nc_eval.py ${opt[$SLURM_ARRAY_TASK_ID]}&

wait  # wait for all ccc_mprun(s) to complete.
