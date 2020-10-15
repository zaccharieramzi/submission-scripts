import time

from fastmri_recon.evaluate.scripts.denoising_eval import evaluate_xpdnet_denoising
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.training_scripts.denoising.generic_train import train_denoiser
import pandas as pd

from jean_zay.submitit.fastmri_reproducible_benchmark.general_submissions import train_eval_grid, eval_grid


job_name = 'denoising_fastmri'
model_name = None
model_size = None
loss = 'compound_mssim'
n_samples = None
n_epochs = 200
contrast = None
model_specs = list(get_model_specs(force_res=True))
if model_name is not None:
    model_specs = [ms for ms in model_specs if ms[0] == model_name]
if model_size is not None:
    model_specs = [ms for ms in model_specs if ms[1] == model_size]


parameter_grid = [
    dict(
        model=(model_fun, kwargs, n_inputs),
        run_id=f'{model_name}_{model_size}_{int(time.time())}',
        contrast=contrast,
        n_epochs=n_epochs,
        n_samples=n_samples,
        loss=loss,
        noise_std=80,
    ) for model_name, model_size, model_fun, kwargs, n_inputs, _, _ in model_specs
]

eval_results = train_eval_grid(
    'denoise',
    train_denoiser,
    evaluate_xpdnet_denoising,
    parameter_grid,
    n_samples_eval=30,
    timeout_train=40,
    n_gpus_train=1,
    timeout_eval=10,
    n_gpus_eval=1,
    to_grid=False,
    noise_std=30,  # just for eval
)

df_results = pd.DataFrame(columns='model_name model_size psnr ssim'.split())

for (name, model_size, _, _, _, _, _), eval_res in zip(model_specs, eval_results):
    df_results = df_results.append(dict(
        model_name=name,
        model_size=model_size,
        psnr=eval_res[0],
        ssim=eval_res[1],
    ), ignore_index=True)

print(df_results)
outputs_file = f'denoising_results_{n_samples}_{loss}.csv'
if model_name is not None:
    outputs_file = f'denoising_results_{n_samples}_{loss}_{model_name}.csv'
df_results.to_csv(outputs_file)