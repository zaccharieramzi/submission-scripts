from learning_wavelets.training_scripts.exact_recon_unet_training import train_unet as train
from learning_wavelets.evaluation_scripts.exact_recon_unet_evaluate import evaluate_unet as evaluate

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'exact_recon_unet'
parameter_grid = {
    'use_bias': [True, False],
    'n_epochs': [500],
    'exact_reconstruction': [True, False],
    'base_n_filters': [64],
}

eval_results = train_eval_grid(
    job_name,
    train,
    evaluate,
    parameter_grid,
    n_samples_eval=None,
    timeout_train=40,
    n_gpus_train=4,
    timeout_eval=4,
    n_gpus_eval=1,
    to_grid=True,
    noise_stds=[0.00001, 5, 15, 20, 25, 30, 50, 55, 60, 75],
    params_to_ignore=[],
    project='learnlets',
)

print(eval_results)
