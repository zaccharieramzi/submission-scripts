from fastmri_recon.evaluate.scripts.xpdnet_eval import evaluate_xpdnet
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.training_scripts.xpdnet_train import train_xpdnet
import pandas as pd

from jean_zay.submitit.fastmri_reproducible_benchmark.general_submissions import train_eval_grid, eval_grid


job_name = 'recon_fastmri'
model_name = None
model_size = None
loss = 'compound_mssim'
n_samples = None
n_epochs = 300
n_primal = 1
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
        multicoil=False,
        n_scales=n_scales,
        res=res,
        n_primal=n_primal,
        contrast=contrast,
        n_epochs=n_epochs,
        n_samples=n_samples,
        af=4,
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

df_results = pd.DataFrame(columns='model_name model_size psnr ssim'.split())

for (name, model_size, _, _, _, _, _), eval_res in zip(model_specs, eval_results):
    df_results = df_results.append(dict(
        model_name=name,
        model_size=model_size,
        psnr=eval_res[0][0],
        psnr_std=eval_res[1][0],
        ssim=eval_res[0][1],
        ssim_std=eval_res[1][1],
    ), ignore_index=True)

print(df_results)
outputs_file = f'recon_results_{n_samples}_{loss}.csv'
if model_name is not None:
    outputs_file = f'recon_results_{n_samples}_{loss}_{model_name}.csv'
df_results.to_csv(outputs_file)
