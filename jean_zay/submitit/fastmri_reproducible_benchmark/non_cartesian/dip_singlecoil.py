from fastmri_recon.evaluate.scripts.nc_dip_eval import evaluate_dip_nc as eval_fun

from jean_zay.submitit.general_submissions import eval_grid

base_params = {
    'af': [4],
    # 'acq_type': ['radial', 'spiral'],
    'acq_type': ['radial'],
    'model_kwargs': [{}],
}


eval_results = eval_grid(
    'dip_sc',
    eval_fun,
    base_params,
    run_ids=None,
    n_samples=10,
    n_gpus=1,
    project='fastmri4',
)
print(eval_results)
