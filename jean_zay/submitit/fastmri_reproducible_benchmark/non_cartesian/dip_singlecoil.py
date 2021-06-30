from fastmri_recon.evaluate.scripts.nc_dip_eval import evaluate_dip_nc as eval_fun

from jean_zay.submitit.general_submissions import eval_grid

base_params = {
    'af': [4, 8],
    'acq_type': ['radial', 'spiral'],
    'contrast': ['CORPD_FBK', 'CORPDFS_FBK'],
    'model_kwargs': [{'bn': True, 'n_up': 5}],
}


eval_results = eval_grid(
    'dip_sc',
    eval_fun,
    base_params,
    run_ids=None,
    n_samples=75,
    n_gpus=1,
    timeout=100,
    project='fastmri4',
)
print(eval_results)
