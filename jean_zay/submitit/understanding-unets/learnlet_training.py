from learning_wavelets.training_scripts.learnlet_training import train_learnlet as train
from learning_wavelets.evaluate_scripts.evaluate_learnlets import evaluate_learnlet as evaluate

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'learnlets_random'
parameter_grid = {
    'random_analysis': [True, False],
    'exact_reconstruction': [True]
}

eval_results = train_eval_grid(
# eval_results = eval_grid(
    job_name,
    train,
    evaluate,
    parameter_grid,
    n_samples_eval=None,
    timeout_train=20,
    n_gpus_train=4,
    timeout_eval=4,
    n_gpus_eval=1,
    # n_samples=200,
    # timeout=10,
    # n_gpus=1,
    to_grid=True,
    noise_stds=[0.00001, 5, 15, 20, 25, 30, 50, 55, 60, 75],
    project='learnlets',
)

print(eval_results)
