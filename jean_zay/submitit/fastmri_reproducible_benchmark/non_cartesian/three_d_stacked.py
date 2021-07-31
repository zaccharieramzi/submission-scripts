from fastmri_recon.evaluate.scripts.nc_eval import evaluate_nc_multinet
from fastmri_recon.training_scripts.nc_train_block import train_ncpdnet

from jean_zay.submitit.general_submissions import train_eval_grid


base_params = {
    'three_d': [True],
    'af': [4],
    'n_filters': [32],
    'n_iter': [10],
    'n_primal': [5],
    'acq_type': ['radial'],
    'loss': ['mse'],
    'dcomp': [True],
    'normalize_image': [False],
    'scale_factor': [1e-2],
    'block_size': [4],
    'use_mixed_precision': [True],
    'epochs_per_block_step': [8],
}

params = [
  dict(block_overlap=[0], n_epochs=[3*8], **base_params),
#   dict(block_overlap=[2], n_epochs=[4*8], **base_params),
]

run_ids = [
    'ncpdnet_3d___bbb_radial_mse_dcomp_1627287586',
    # '',
]

eval_results = train_eval_grid(
    '3d_nc_stacked',
    train_ncpdnet,
    evaluate_nc_multinet,
    params,
    # run_ids=run_ids,
    n_gpus_train=1,
    timeout_train=100,
    n_gpus_eval=1,
    n_samples_eval=100,
    params_to_ignore=['use_mixed_precision', 'scale_factor', 'epochs_per_block_step', 'block_size', 'block_overlap'],
    checkpoints_train=7,
    resume_checkpoint=1,
    resume_run_run_ids=run_ids,
    project='fastmri4',
)
print(eval_results)
