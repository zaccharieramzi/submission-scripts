from mdeq_lib.training_scripts.noise_estimation_train import noise_estimator_train
from mdeq_lib.evaluate.scripts.noise_estimation_eval import eval_noise_estimator

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'noise_estimation'
parameters = dict(
    n_val=[20],
    grad_clipping=[100],
    n_epochs=[5000],
    num_blocks=[1, 3, 5],
)

train_eval_grid(
    job_name,
    noise_estimator_train,
    eval_noise_estimator,
    parameters,
    # run_ids=[None],
    to_grid=True,
    timeout_train=20,
    n_gpus_train=1,
    timeout_eval=2,
    n_gpus_eval=1,
    # timeout=10,
    # n_gpus=1,
    project='mdeq',
    params_to_ignore=['grad_clipping', 'n_val'],
    val_noise_powers=[1e-3, 1e-2, 1e-1, 10, 50],
    no_force_32=True,
)
