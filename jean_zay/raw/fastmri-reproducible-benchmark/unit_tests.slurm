#!/bin/bash
#SBATCH --job-name=fastmi_unit_test     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --qos=qos_gpu-dev        # le calcul va etre long
#SBATCH --partition=archive
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=2:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%j.out # nom du fichier de sortie
#SBATCH --error=%x_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-8

set -x

opt[0]="fastmri_recon/tests/data"
opt[1]="fastmri_recon/tests/models"
opt[2]="fastmri_recon/tests/training_scripts/denoising"
opt[3]="fastmri_recon/tests/training_scripts/multi_coil"
opt[4]="fastmri_recon/tests/training_scripts/dealiasing_train_test.py"
opt[5]="fastmri_recon/tests/training_scripts/ncpdnet_train_test.py"
opt[6]="fastmri_recon/tests/training_scripts/unet_nc_train_test.py"
opt[7]="fastmri_recon/tests/training_scripts/updnet_train_test.py"
opt[8]="fastmri_recon/tests/training_scripts/xpdnet_train_test.py"

cd $WORK/fastmri-reproducible-benchmark

srun pytest -s ${opt[$SLURM_ARRAY_TASK_ID]}
