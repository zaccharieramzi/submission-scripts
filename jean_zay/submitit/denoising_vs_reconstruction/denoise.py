import time

from denoising_recon.evaluate.evaluation import evaluate
from denoising_recon.models.denoisers.proposed_params import get_model_specs
from denoising_recon.training.train import train
import pandas as pd

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'denoising_celeba'
model_name = None
model_size = 'small'
loss = 'mae'
bn = False
non_linearity = 'relu'
n_samples = None
n_epochs = 5
grey = True
noise_std = 1
noise_conditionning = True
additional_info = f'_ns{noise_std}'
if not grey:
    additional_info += '_color'
if loss != 'mse':
    additional_info += f'_{loss}'
if noise_conditionning:
    additional_info += '_noisecon'
model_specs = list(get_model_specs(grey=grey, mod_kwargs=dict(bn=bn, non_linearity=non_linearity)))
if model_name is not None:
    model_specs = [ms for ms in model_specs if ms[0] == model_name]
if model_size is not None:
    model_specs = [ms for ms in model_specs if ms[1] == model_size]
if noise_conditionning:
    # FocNet doesn't support noise conditionning
    model_specs = [ms for ms in model_specs if not 'FocNet' in ms[0]]


parameter_grid = [
    dict(
        is_denoising=True,
        model_fun=model_fun,
        model_kwargs=kwargs,
        noise_conditionning=noise_conditionning,
        grey=grey,
        run_id=f'{model_name}_{model_size}_denoising{additional_info}_{int(time.time())}',
        n_epochs=n_epochs,
        n_samples=n_samples,
        loss=loss,
        noise_power_spec=noise_std,
        target_size=(128, 128),
    ) for model_name, model_size, model_fun, kwargs, _ in model_specs
]

eval_results = train_eval_grid(
# eval_results = eval_grid(
    job_name,
    train,
    evaluate,
    parameter_grid,
    n_samples_eval=500,
    timeout_train=80,
    n_gpus_train=1,
    timeout_eval=4,
    n_gpus_eval=1,
    # n_samples=200,
    # timeout=10,
    # n_gpus=1,
    to_grid=False,
    noise_power_spec=1,  # just for eval
    project='denoise_vs_recon',
)

df_results = pd.DataFrame(columns='model_name model_size psnr ssim'.split())

for (name, model_size, _, _, _), eval_res in zip(model_specs, eval_results):
    df_results = df_results.append(dict(
        model_name=name,
        model_size=model_size,
        psnr=eval_res[0],
        ssim=eval_res[1],
    ), ignore_index=True)

print(df_results)
outputs_file = f'denoising_results_{n_samples}{additional_info}.csv'
if model_name is not None:
    outputs_file = f'denoising_results_{n_samples}{additional_info}_{model_name}.csv'
df_results.to_csv(outputs_file)
