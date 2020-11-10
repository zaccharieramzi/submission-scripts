from fastmri_recon.evaluate.scripts.nc_eval import evaluate_nc_multinet
from fastmri_recon.training_scripts.nc_train import train_ncnet_multinet

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


base_params = {
    'three_d': [True],
    'n_epochs': [15],
    'af': [4],
    'n_filters': [4],
    'n_iter': [4],
    'n_primal': [2],
    'acq_type': ['radial_stacks', 'spiral_stacks'],
    'loss': ['mse'],
}

params = [
  dict(dcomp=[True], normalize_image=[False], model=['unet', 'pdnet'], **base_params),
  dict(dcomp=[False], normalize_image=[True], model=['pdnet'], **base_params),
]

train_eval_grid('3d_nc', train_ncnet_multinet, evaluate_nc_multinet, params, n_samples_eval=30)
