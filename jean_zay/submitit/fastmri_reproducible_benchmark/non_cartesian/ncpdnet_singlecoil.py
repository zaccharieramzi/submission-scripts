from fastmri_recon.evaluate.scripts.nc_eval import evaluate_ncpdnet as eval_fun
from fastmri_recon.training_scripts.nc_train import train_ncpdnet as train_fun

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid

base_params = {
    'n_epochs': [100],
    'af': [4],
    'acq_type': ['radial', 'spiral'],
    'loss': ['compound_mssim'],
}

params = [
  dict(dcomp=[True], normalize_image=[False], **base_params),
  dict(dcomp=[False], normalize_image=[True], **base_params),
]

eval_results = train_eval_grid(
    'nc_pdnet',
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
