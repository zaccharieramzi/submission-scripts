from mdeq_lib.training_scripts.denoise_train import train_dncnn_denoising
from mdeq_lib.evaluate.scripts.denoise_eval import eval_dncnn_denoising

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'dncnn_denoise'
parameters = [
    dict(
        n_val=20,
        grad_clipping=100,
        fixed_lr=True,
        loss='mae',
        n_epochs=5000,
        inpaint=True,
        train_noise_power_range=(5, 5),
        val_noise_power_range=(5, 5),
        inpainting_train_range=(70, 70),
        inpainting_val_range=(70, 70),
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
    timeout_eval=2,
    n_gpus_eval=1,
    # timeout=10,
    # n_gpus=1,
    project='mdeq',
    params_to_ignore=[
        'grad_clipping',
        'n_val',
        'train_noise_power_range',
        'val_noise_power_range',
        'inpainting_train_range',
        'inpainting_val_range',
    ],
    val_noise_powers=[5],
    inpainting_range=(70, 70),
    no_force_32=True,
)
