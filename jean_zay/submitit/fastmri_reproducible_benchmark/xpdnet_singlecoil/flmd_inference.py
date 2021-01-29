from fastmri_recon.evaluate.scripts.xpdnet_inference import xpdnet_inference
from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.proposed_params import get_model_specs

from jean_zay.submitit.general_submissions import infer_grid


job_name = 'xpdnet_flmd_inference'
n_samples = None
n_epochs = 300
n_primal = 5
contrast = None
refine_smaps = True
refine_big = False
n_dual = 3
primal_only = True
multiscale_kspace_learning = False
n_dual_filters = 8
n_iter_per_model_size = {
    'big': 7,
}
use_mixed_precision = False
model_specs = list(get_model_specs(n_primal=n_primal))

run_ids = [
    'xpdnet_singlecoil__af8_i7_compound_mssim_rf_sm_UnetMultiDomainbig_1610912245',
    'xpdnet_singlecoil__af4_i7_compound_mssim_rf_sm_UnetMultiDomainbig_1610912352',
]

parameter_grid = [
    dict(
        model_fun=model_fun,
        model_kwargs=kwargs,
        exp_id='xpdnet_flmd',
        multicoil=False,
        n_scales=n_scales,
        res=res,
        n_primal=n_primal,
        contrast=contrast,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_samples=n_samples,
        refine_smaps=refine_smaps,
        refine_big=refine_big,
        af=8,
        n_dual=n_dual,
        n_iter=n_iter_per_model_size[model_size],
        primal_only=primal_only,
        n_dual_filters=n_dual_filters,
    ) for model_name, model_size, model_fun, kwargs, _, n_scales, res in model_specs
    if model_size == 'big' and 'MWCNN' not in model_name
] + [
    dict(
        model_fun=model_fun,
        model_kwargs=kwargs,
        exp_id='xpdnet_flmd',
        multicoil=False,
        n_scales=n_scales,
        res=res,
        n_primal=n_primal,
        contrast=contrast,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_samples=n_samples,
        refine_smaps=refine_smaps,
        refine_big=refine_big,
        af=4,
        n_dual=n_dual,
        n_iter=n_iter_per_model_size[model_size],
        primal_only=primal_only,
        n_dual_filters=n_dual_filters,
    ) for model_name, model_size, model_fun, kwargs, _, n_scales, res in model_specs
    if model_size == 'big' and 'MWCNN' not in model_name
]


infer_grid(
    job_name,
    xpdnet_inference,
    parameter_grid,
    run_ids=run_ids,
    timeout=10,
    n_gpus=4,
    to_grid=False,
)
