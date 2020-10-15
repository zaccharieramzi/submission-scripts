#!/bin/bash
#SBATCH --job-name=annealed_langevin     # nom du job
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
#SBATCH --output=annealed_langevin%A_%a.out # nom du fichier de sortie
#SBATCH --error=annealed_langevin%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)

set -x
cd $WORK/score-estimation-comparison

#!/bin/bash
module purge
conda deactivate fastmri-tf-2.1.0
module load tensorflow-gpu/py3/2.3.0

export FASTMRI_DATA_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH/nsec_3scales
export FIGURES_DIR=$SCRATCH/nsec_figures/annealed_langevin_sn5_3scales

srun python ./nsec/mri/annealed_langevin_reconstruction.py -n 30000 -b 2 -c CORPD_FBK -sn 5. -is 64 -nps 50. -sc 3 -h -nr 10
