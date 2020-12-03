from learning_wavelets.training_scripts.dncnn_training import train_dncnn as train
from learning_wavelets.evaluate_scripts.evaluate_dncnn import evaluate_dncnn as evaluate

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'dncnn'
parameter_grid = {
    'bn': [True, False],
    'n_epochs': [100],
    'steps_per_epoch': [3000],
}

eval_results = train_eval_grid(
# run_ids = ['dncnn_bn_1606936039', 'dncnn1606942875']
# eval_results = eval_grid(
    job_name,
    train,
    evaluate,
    parameter_grid,
    # run_ids=run_ids,
    n_samples_eval=None,
    timeout_train=20,
    n_gpus_train=4,
    timeout_eval=4,
    n_gpus_eval=1,
    # n_samples=None,
    # timeout=10,
    # n_gpus=1,
    to_grid=True,
    noise_stds=[0.00001, 5, 15, 20, 25, 30, 50, 55, 60, 75],
    params_to_ignore=['steps_per_epoch'],
    project='learnlets',
)

print(eval_results)
