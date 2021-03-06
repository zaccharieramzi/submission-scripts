#!/bin/bash
#SBATCH --job-name=3scales_map_recon_mri_hd     # nom du job
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
#SBATCH --output=3scales_map_recon_mri_hd%A_%a.out # nom du fichier de sortie
#SBATCH --error=3scales_map_recon_mri_hd%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-6

set -x
cd $WORK/score-estimation-comparison

#!/bin/bash
module purge
conda deactivate fastmri-tf-2.1.0
module load tensorflow-gpu/py3/2.3.0

opt[0]='0.1'
opt[1]='0.5'
opt[2]='1.0'
opt[3]='2.0'
opt[4]='5.0'
opt[5]='10.0'
opt[6]='0.'

export FASTMRI_DATA_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/nsec_3scales/
export FIGURES_DIR=$SCRATCH/nsec_figures/map_recon_3scales_hd_sn${opt[$SLURM_ARRAY_TASK_ID]}/

srun python ./nsec/mri/map_reconstruction.py -n 1000000 -nps 50.0 -b 2 -c CORPD_FBK -h -e 0.0000001 -sn ${opt[$SLURM_ARRAY_TASK_ID]} -sc 3
