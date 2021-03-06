#!/bin/bash
#SBATCH --job-name=updnet_brain_infer     # nom du job
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
#SBATCH --output=updnet_brain_infer%A_%a.out # nom du fichier de sortie
#SBATCH --error=updnet_brain_infer%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-9
#SBATCH --dependency=afterok:167442

set -x
cd $WORK/fastmri-reproducible-benchmark

. ./submission_scripts_jean_zay/env_config.sh

export EXP_ID="updnet_v1"

export RUN_ID_AF4_AXFLAIR="updnet_sense_brain__af4_AXFLAIR_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF4_AXT1POST="updnet_sense_brain__af4_AXT1POST_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF4_AXT1PRE="updnet_sense_brain__af4_AXT1PRE_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF4_AXT1="updnet_sense_brain__af4_AXT1_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF4_AXT2="updnet_sense_brain__af4_AXT2_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF8_AXFLAIR="updnet_sense_brain__af8_AXFLAIR_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF8_AXT1POST="updnet_sense_brain__af8_AXT1POST_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF8_AXT1PRE="updnet_sense_brain__af8_AXT1PRE_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF8_AXT1="updnet_sense_brain__af8_AXT1_i20_compound_mssim_rf_sm_1599121045"
export RUN_ID_AF8_AXT2="updnet_sense_brain__af8_AXT2_i20_compound_mssim_rf_sm_1599121045"

# AF4 infer
opt[0]="-x ${EXP_ID} -a 4 -b -i 20 -rfs -r ${RUN_ID_AF4_AXFLAIR} -e 10 -c AXFLAIR"
opt[1]="-x ${EXP_ID} -a 4 -b -i 20 -rfs -r ${RUN_ID_AF4_AXT1POST} -e 10 -c AXT1POST"
opt[2]="-x ${EXP_ID} -a 4 -b -i 20 -rfs -r ${RUN_ID_AF4_AXT1PRE} -e 10 -c AXT1PRE"
opt[3]="-x ${EXP_ID} -a 4 -b -i 20 -rfs -r ${RUN_ID_AF4_AXT1} -e 10 -c AXT1"
opt[4]="-x ${EXP_ID} -a 4 -b -i 20 -rfs -r ${RUN_ID_AF4_AXT2} -e 10 -c AXT2"
# AF8 infer
opt[5]="-x ${EXP_ID} -a 8 -b -i 20 -rfs -r ${RUN_ID_AF8_AXFLAIR} -e 10 -c AXFLAIR"
opt[6]="-x ${EXP_ID} -a 8 -b -i 20 -rfs -r ${RUN_ID_AF8_AXT1POST} -e 10 -c AXT1POST"
opt[7]="-x ${EXP_ID} -a 8 -b -i 20 -rfs -r ${RUN_ID_AF8_AXT1PRE} -e 10 -c AXT1PRE"
opt[8]="-x ${EXP_ID} -a 8 -b -i 20 -rfs -r ${RUN_ID_AF8_AXT1} -e 10 -c AXT1"
opt[9]="-x ${EXP_ID} -a 8 -b -i 20 -rfs -r ${RUN_ID_AF8_AXT2} -e 10 -c AXT2"

srun python ./fastmri_recon/evaluate/scripts/updnet_sense_inference.py ${opt[$SLURM_ARRAY_TASK_ID]}&

wait  # wait for all ccc_mprun(s) to complete.
