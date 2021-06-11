from fastmri_recon.evaluate.scripts.nc_dip_eval import evaluate_dip_nc as eval_fun

from jean_zay.submitit.general_submissions import eval_grid

base_params = {
    'af': [4],
    'acq_type': ['radial', 'spiral'],
    'model_kwargs': [{}],
}


eval_results = eval_grid(
    'nc_pdnet',
    eval_fun,
    base_params,
    run_ids=None,
    n_samples=100,
    n_gpus=1,
)
print(eval_results)
