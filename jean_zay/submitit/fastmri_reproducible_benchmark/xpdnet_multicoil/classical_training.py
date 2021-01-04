from fastmri_recon.evaluate.scripts.xpdnet_eval import evaluate_xpdnet
from fastmri_recon.evaluate.scripts.xpdnet_inference import xpdnet_inference
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.training_scripts.xpdnet_train import train_xpdnet

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid, infer_grid


job_name = 'classical_xpdnet'
model_name = 'MWCNN'
model_size = 'medium'
loss = 'compound_mssim'
lr = 1e-4
batch_size = None
n_samples = None
n_epochs = 400
n_primal = 5
contrast = None
refine_smaps = True
refine_big = True
n_dual = 2
primal_only = True
multiscale_kspace_learning = False
n_dual_filters = 8
n_iter = 25
use_mixed_precision = False
model_specs = list(get_model_specs(force_res=False, n_primal=n_primal))
if model_name is not None:
    model_specs = [ms for ms in model_specs if ms[0] == model_name]
if model_size is not None:
    model_specs = [ms for ms in model_specs if ms[1] == model_size]

parameter_grid = [
    dict(
        model_fun=model_fun,
        model_kwargs=kwargs,
        model_size=model_size,
        multicoil=True,
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
        loss=loss,
        lr=lr,
        mask_type='random',
        n_dual=n_dual,
        n_iter=n_iter,
        primal_only=primal_only,
        n_dual_filters=n_dual_filters,
        multiscale_kspace_learning=multiscale_kspace_learning,
        use_mixed_precision=use_mixed_precision,
    ) for _, model_size, model_fun, kwargs, _, n_scales, res in model_specs
] + [
    dict(
        model_fun=model_fun,
        model_kwargs=kwargs,
        model_size=model_size,
        multicoil=True,
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
        loss=loss,
        lr=lr,
        mask_type='random',
        n_dual=n_dual,
        n_iter=n_iter,
        primal_only=primal_only,
        n_dual_filters=n_dual_filters,
        multiscale_kspace_learning=multiscale_kspace_learning,
        use_mixed_precision=use_mixed_precision,
    ) for _, model_size, model_fun, kwargs, _, n_scales, res in model_specs
]

# eval_results, run_ids = train_eval_grid(
run_ids = [
    'xpdnet_sense__af8_i25_compound_mssim_rf_smb_MWCNNmedium_1608723151',
    'xpdnet_sense__af4_i25_compound_mssim_rf_smb_MWCNNmedium_1608723151',
]

eval_results = eval_grid(
    job_name,
    # train_xpdnet,
    evaluate_xpdnet,
    parameter_grid,
    run_ids=run_ids,
    # n_samples_eval=50,
    # timeout_train=70,
    # n_gpus_train=batch_size if batch_size else 1,
    # timeout_eval=10,
    # n_gpus_eval=1,
    n_samples=50,
    timeout=10,
    n_gpus=1,
    to_grid=False,
    # return_run_ids=True,
    # checkpoints_train=4,
    params_to_ignore=['batch_size', 'use_mixed_precision'],
)

print(eval_results)

infer_grid(
    job_name,
    xpdnet_inference,
    parameter_grid,
    run_ids=run_ids,
    timeout=10,
    n_gpus=4,
    to_grid=False,
    params_to_ignore=['mask_type', 'multicoil', 'batch_size', 'use_mixed_precision'],
)
