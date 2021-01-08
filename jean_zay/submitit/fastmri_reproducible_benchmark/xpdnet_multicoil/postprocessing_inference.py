from fastmri_recon.evaluate.scripts.postprocess_inference import postproc_inference
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs

from jean_zay.submitit.general_submissions import infer_grid


job_name = 'post_process_infer'
run_ids = [
    'vnet_postproc__af4_compound_mssim_bf4_sc3_lrelu_1610015187',
    'vnet_postproc__af8_compound_mssim_bf4_sc3_lrelu_1610015187',
]
brain = False
n_epochs = 190
base_n_filters = 4
n_scales = 3
non_linearity = 'lrelu'

parameter_grid = dict(
    recon_path=['xpdnet_v4'],
    brain=[brain],
    n_epochs=[n_epochs],
    af=[4, 8],
    base_n_filters=[base_n_filters],
    n_scales=[n_scales],
    non_linearity=[non_linearity],
)

infer_grid(
    job_name,
    postproc_inference,
    parameter_grid,
    run_ids=run_ids,
    timeout=20,
    n_gpus=1,
    to_grid=True,
)
