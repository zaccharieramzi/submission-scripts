from distributed_mri_reconstruction.evaluate.eval import evaluate
from distributed_mri_reconstruction.train import train

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


base_params = {
    'n_epochs': [15],
    'af': [4],
    'n_filters': [32],
    'n_iter': [6],
    'n_primal': [3],
    'acq_type': ['radial_stacks', 'spiral_stacks'],
    'loss': ['mse'],
}

train_eval_grid('3d_nc_mesh', train, evaluate, base_params, n_samples_eval=30)
