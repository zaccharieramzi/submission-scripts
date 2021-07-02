from fastmri_recon.evaluate.scripts.nc_dip_eval import evaluate_dip_nc as eval_fun

from jean_zay.submitit.general_submissions import eval_grid

base_params = {
    'af': [4, 8],
    # 'af': [4],
    'acq_type': ['radial', 'spiral'],
    # 'acq_type': ['radial'],
    'contrast': ['CORPD_FBK', 'CORPDFS_FBK'],
    # 'contrast': ['CORPD_FBK'],
    'model_kwargs': [{'bn': True, 'n_up': 5}],
    'multicoil': [True],
}


eval_results = eval_grid(
    'dip_mc',
    eval_fun,
    base_params,
    run_ids=None,
    n_samples=65,
    n_gpus=1,
    timeout=100,
    project='fastmri4',
)
print(eval_results)
