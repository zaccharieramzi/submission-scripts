#!/bin/bash
#SBATCH --job-name=denoiser_score_est_train     # nom du job
#SBATCH --ntasks=1                   # nombre de tâche MPI
#SBATCH --ntasks-per-node=1          # nombre de tâche MPI par noeud
#SBATCH --gres=gpu:1                 # nombre de GPU à réserver par nœud
#SBATCH --cpus-per-task=10           # nombre de coeurs à réserver par tâche
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread         # on réserve des coeurs physiques et non logiques
#SBATCH --distribution=block:block   # on épingle les tâches sur des coeurs contigus
#SBATCH --time=60:00:00              # temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=denoiser_score_est_train%A_%a.out # nom du fichier de sortie
#SBATCH --error=denoiser_score_est_train%A_%a.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-5

set -x
cd $WORK/score-estimation-comparison

#!/bin/bash
module purge
conda deactivate fastmri-tf-2.1.0
module load tensorflow-gpu/py3/2.3.0

export FASTMRI_DATA_DIR=$SCRATCH/
export CHECKPOINTS_DIR=$SCRATCH//nsec_complex

opt[0]="-sn 1."
opt[1]="-sn 0.5"
opt[2]="-sn 0.1"
opt[3]=""
opt[4]="-sn 5."
opt[5]="-sn 10."

srun python ./nsec/mri/denoiser_training.py ${opt[$SLURM_ARRAY_TASK_ID]} -n 100000 -c CORPD_FBK -nps 50 -lr 0.0001
