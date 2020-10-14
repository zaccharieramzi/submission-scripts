#!/bin/bash
#SBATCH --job-name=tempered_sampling     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=20:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=tempered_sampling%A_%a.out # nom du fichier de sortie
#SBATCH --error=tempered_sampling%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-4
#SBATCH --dependency=afterok:413082

set -x
cd $WORK/score-estimation-comparison

#!/bin/bash
module purge
conda deactivate fastmri-tf-2.1.0
module load tensorflow-gpu/py3/2.3.0

opt[0]="2.0"
opt[1]="0.5"
opt[2]="0."
opt[3]="1.0"
opt[4]="5.0"

export FASTMRI_DATA_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/nsec_nofc_3scales_norm
export FIGURES_DIR=$SCRATCH/nsec_figures/tempered_sampling_norm_nofc_3scales_sn${opt[$SLURM_ARRAY_TASK_ID]}

srun python ./nsec/mri/tempered_sampling_reconstruction.py -n 3000 -b 2 -c CORPD_FBK -sn ${opt[$SLURM_ARRAY_TASK_ID]} -nps 0.5 --no-fcon -sc 3 -nr 7 -e 10 -dcs 0.01 -si 1 -sf 10000
