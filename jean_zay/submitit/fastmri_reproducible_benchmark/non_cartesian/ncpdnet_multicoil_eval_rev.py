from fastmri_recon.evaluate.scripts.nc_eval import evaluate_ncpdnet as eval_fun
from fastmri_recon.training_scripts.nc_train import train_ncpdnet as train_fun

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid

base_params = {
    'n_epochs': [100],
    'af': [4],
    'acq_type': ['radial', 'spiral'],
    'loss': ['compound_mssim'],
    'multicoil': [True],
    'refine_smaps': [True],
}

params = [
  dict(dcomp=[True], normalize_image=[False], **base_params),
]

run_ids = [
    'ncpdnet_sense___rfs_spiral_compound_mssim_dcomp_1611913984',
    'ncpdnet_sense___rfs_radial_compound_mssim_dcomp_1611913984',
]

eval_results = eval_grid(
    'nc_pdnet_mc_rev',
    eval_fun,
    params,
    run_ids=run_ids,
    n_samples=100,
    n_gpus=3,
    project='fastmri4',
)
print(eval_results)
