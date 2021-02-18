from mdeq_lib.training_scripts.denoise_train import train_mdeq_denoising
from mdeq_lib.evaluate.scripts.denoise_eval import eval_mdeq_denoising

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'mdeq_denoise'
parameters = [dict(
    n_val=20,
    grad_clipping=100.,
    loss='mae',
    n_epochs=2000,
    fixed_lr=True,
    use_res=True,
    use_bn=True,
)]

train_eval_grid(
    job_name,
    train_mdeq_denoising,
    eval_mdeq_denoising,
    parameters,
    to_grid=False,
    timeout_train=20,
    n_gpus_train=1,
    timeout_eval=10,
    n_gpus_eval=1,
    project='mdeq',
    params_to_ignore=['grad_clipping', 'n_val'],
    val_noise_powers=[15, 25, 50],
)
