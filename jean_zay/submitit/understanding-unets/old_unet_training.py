from learning_wavelets.training_scripts.exact_recon_old_unet_training import train_old_unet as train
from learning_wavelets.evaluation_scripts.exact_recon_old_unet_evaluate import evaluate_old_unet as evaluate

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'exact_recon_old_unet'
parameter_grid = {
    'use_bias': [False],
    'n_epochs': [500],
    'exact_recon': [True, False],
    'base_n_filters': [64],
}

eval_results = train_eval_grid(
# eval_results = eval_grid(
    job_name,
    train,
    evaluate,
    parameter_grid,
    # run_ids=run_ids,
    n_samples_eval=None,
    # n_samples=None,
    timeout_train=6,
    n_gpus_train=4,
    timeout_eval=4,
    # timeout=4,
    n_gpus_eval=1,
    # n_gpus=1,
    to_grid=True,
    noise_std_test=[0.00001, 5, 15, 20, 25, 30, 50, 55, 60, 75, 85, 95, 100],
    params_to_ignore=[],
    project='learnlets',
    force_partition='gpu_p2',
)

print(eval_results)
