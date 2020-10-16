from fastmri_recon.evaluate.scripts.xpdnet_eval import evaluate_xpdnet
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.training_scripts.xpdnet_train import train_xpdnet

from jean_zay.submitit.fastmri_reproducible_benchmark.general_submissions import train_eval_grid, eval_grid


job_name = 'fastmri_recon_equidistant_mc'
model_name = 'MWCNN'
model_size = 'medium'
loss = 'compound_mssim'
n_samples = None
n_epochs = 300
n_primal = 5
contrast = 'CORPD_FBK'
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
        n_epochs=n_epochs,
        n_samples=n_samples,
        af=8,
        loss=loss,
    ) for _, model_size, model_fun, kwargs, _, n_scales, res in model_specs
]

eval_results = train_eval_grid(
    job_name,
    train_xpdnet,
    evaluate_xpdnet,
    parameter_grid,
    n_samples_eval=100,
    timeout_train=60,
    n_gpus_train=1,
    timeout_eval=10,
    n_gpus_eval=1,
    to_grid=False,
)

print(eval_results)
