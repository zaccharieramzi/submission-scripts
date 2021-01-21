from fastmri_recon.evaluate.scripts.nc_eval import evaluate_unet as eval_fun
from fastmri_recon.training_scripts.nc_train import train_unet_nc as train_fun

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid

params = {
    'n_epochs': [100],
    'af': [4],
    'acq_type': ['radial', 'spiral'],
    'loss': ['compound_mssim'],
    'dcomp': [True],
    'multicoil': [True],
}

# run_ids = [
    # 'unet_singlecoil___radial_compound_mssim_dcomp_1610911070',
    # 'unet_singlecoil___spiral_compound_mssim_dcomp_1610911070',
# ]

eval_results = train_eval_grid(
    'nc_unet_mc',
    # train_fun,
    eval_fun,
    params,
    run_ids=run_ids,
    n_gpus_train=1,
    timeout_train=100,
    n_gpus_eval=1,
    n_samples_eval=100,
    # n_samples=100,
    # n_gpus=1,
)
print(eval_results)
