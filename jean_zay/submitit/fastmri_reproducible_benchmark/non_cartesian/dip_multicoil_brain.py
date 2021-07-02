from fastmri_recon.evaluate.scripts.nc_dip_eval import evaluate_dip_nc as eval_fun

from jean_zay.submitit.general_submissions import eval_grid

base_params = {
    'af': [4],
    # 'acq_type': ['radial', 'spiral'],
    'acq_type': ['radial'],
    # 'contrast': ['AXT1', 'AXT1POST', 'AXT1PRE', 'AXT2', 'AXFLAIR'],
    'contrast': ['AXT1'],
    'model_kwargs': [{'bn': True, 'n_up': 5}],
    'multicoil': [True],
    'brain': [True],
}


eval_results = eval_grid(
    'dip_mc_brain',
    eval_fun,
    base_params,
    run_ids=None,
    n_samples=1,
    n_gpus=1,
    timeout=5,
    project='fastmri4',
)
print(eval_results)
