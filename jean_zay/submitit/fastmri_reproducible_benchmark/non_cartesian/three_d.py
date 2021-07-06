from fastmri_recon.evaluate.scripts.nc_eval import evaluate_nc_multinet
from fastmri_recon.training_scripts.nc_train import train_ncnet_multinet

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


base_params = {
    'three_d': [True],
    'n_epochs': [8],
    'af': [4],
    'n_filters': [16],
    'n_iter': [6],
    'n_primal': [2],
    'acq_type': ['radial'],
    'loss': ['mse'],
    'dcomp': [True],
    'normalize_image': [False],
    'scale_factor': [1e-2],
}

params = [
  dict(use_mixed_precision=[True], model=['pdnet'], **base_params),
  dict(use_mixed_precision=[False], model=['unet'], **base_params),
]


eval_results = train_eval_grid(
    '3d_nc',
    train_ncnet_multinet,
    evaluate_nc_multinet,
    params,
    # run_ids=run_ids,
    n_gpus_train=1,
    timeout_train=40,
    n_gpus_eval=1,
    n_samples_eval=100,
    params_to_ignore=['use_mixed_precision', 'scale_factor'],
    checkpoints_train=7,
    # resume_checkpoint=4,
    # resume_run_run_ids=run_ids,
    project='fastmri4',
)
print(eval_results)
