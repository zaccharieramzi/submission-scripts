from mdeq_lib.training_scripts.denoise_train import train_dncnn_denoising
from mdeq_lib.evaluate.scripts.denoise_eval import eval_dncnn_denoising

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'dncnn_denoise'
parameters = [
    dict(
        n_val=20,
        network_size='SMALL',
        grad_clipping=100.,
        state_residual=True,
    )
]
train_eval_grid(
    job_name,
    train_dncnn_denoising,
    eval_dncnn_denoising,
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
