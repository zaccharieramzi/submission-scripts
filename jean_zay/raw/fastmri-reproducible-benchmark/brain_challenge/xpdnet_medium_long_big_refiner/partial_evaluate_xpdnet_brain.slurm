#!/bin/bash
#SBATCH --job-name=xpdnet_brain_eval_partial     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:2                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=20           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=xpdnet_brain_eval_partial%A_%a.out # nom du fichier de sortie
#SBATCH --error=xpdnet_brain_eval_partial%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-9
#SBATCH --dependency=afterok:384652

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

export RUN_ID_AF4="xpdnet_sense_brain__af4_i25_compound_mssim_rf_smb_MWCNNmedium_1601987069"
export RUN_ID_AF8="xpdnet_sense_brain__af8_i25_compound_mssim_rf_smb_MWCNNmedium_1601987069"

# AF4 eval
opt[0]="-a 4 -r ${RUN_ID_AF4} -c AXFLAIR"
opt[1]="-a 4 -r ${RUN_ID_AF4} -c AXT1POST"
opt[2]="-a 4 -r ${RUN_ID_AF4} -c AXT1PRE"
opt[3]="-a 4 -r ${RUN_ID_AF4} -c AXT1"
opt[4]="-a 4 -r ${RUN_ID_AF4} -c AXT2"
# AF8 eval
opt[5]="-a 8 -r ${RUN_ID_AF8} -c AXFLAIR"
opt[6]="-a 8 -r ${RUN_ID_AF8} -c AXT1POST"
opt[7]="-a 8 -r ${RUN_ID_AF8} -c AXT1PRE"
opt[8]="-a 8 -r ${RUN_ID_AF8} -c AXT1"
opt[9]="-a 8 -r ${RUN_ID_AF8} -c AXT2"

srun python ./fastmri_recon/evaluate/scripts/xpdnet_eval.py ${opt[$SLURM_ARRAY_TASK_ID]} -i 25 -b -eqf -n 30 -rfsb -m MWCNN -s medium -e 100&

wait  # wait for all ccc_mprun(s) to complete.
