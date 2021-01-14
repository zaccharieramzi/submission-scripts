from fastmri_recon.evaluate.scripts.nc_eval import evaluate_unet as eval_fun
from fastmri_recon.training_scripts.nc_train import train_unet_nc as train_fun

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid

params = {
    'n_epochs': [100],
    'af': [4],
    'acq_type': ['radial', 'spiral'],
    'loss': ['compound_mssim'],
    'dcomp': [True],
}

eval_results = train_eval_grid(
    'nc_unet',
    train_fun,
    eval_fun,
    params,
    n_gpus_train=1,
    timeout_train=100,
    n_gpus_eval=1,
    n_samples_eval=100,
    checkpoints_train=0,
    resume_checkpoint=0,
    resume_run_run_ids=None,
)
print(eval_results)
