from mdeq_lib.training_scripts.denoise_train import train_ipdeq_denoising
from mdeq_lib.evaluate.scripts.denoise_eval import eval_ipdeq_denoising

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'ipdeq_inpaint'
parameters = dict(
    n_val=[20],
    grad_clipping=[100.],
    state_residual=[False],
    num_blocks=[20],
    with_dc=[True],
    loss=['mae'],
    n_epochs=[5000],
    fixed_lr=[True],
    unrolled_supp_validation=[True],
    debug_deq=[True],
    lr_dc=['trainable'],
    inpaint=[True],
    train_noise_power_range=[(5, 5)],
    val_noise_power_range=[(5, 5)],
    inpainting_train_range=[(70, 70)],
    inpainting_val_range=[(70, 70)],
)

train_eval_grid(
    job_name,
    train_ipdeq_denoising,
    eval_ipdeq_denoising,
    parameters,
    to_grid=True,
    timeout_train=1,
    n_gpus_train=1,
    timeout_eval=2,
    n_gpus_eval=1,
    project='mdeq',
    params_to_ignore=[
        'grad_clipping',
        'n_val',
        'train_noise_power_range',
        'val_noise_power_range',
        'inpainting_train_range',
        'inpainting_val_range',
        'debug_deq',
    ],
    val_noise_powers=[5],
    inpainting_range=(70, 70),
    max_steps=100,
)
