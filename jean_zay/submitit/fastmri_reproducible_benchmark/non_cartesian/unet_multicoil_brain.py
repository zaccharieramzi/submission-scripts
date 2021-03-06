from fastmri_recon.evaluate.scripts.nc_eval import evaluate_unet as eval_fun

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid

params = {
    'n_epochs': [100],
    'af': [4],
    'acq_type': ['radial', 'spiral'],
    'loss': ['compound_mssim'],
    'dcomp': [True],
    'multicoil': [True],
    'brain': [True],
}

run_ids = [
    'unet_mc___radial_compound_mssim_dcomp_1611915508',
    'unet_mc___spiral_compound_mssim_dcomp_1611915508',
]

eval_results = eval_grid(
    'nc_unet_mc_brain',
    eval_fun,
    params,
    run_ids=run_ids,
    n_samples=250,
    n_gpus=3,
    timeout=20,
    project='fastmri4',
)
print(eval_results)
