from learning_wavelets.evaluate_tmp.dncnn_eval import evaluate_dncnn
from learning_wavelets.evaluate_tmp.results_to_csv import results_to_csv
from learning_wavelets.training.scripts.dncnn_training import train_dncnn

from generic_dask import train_eval_parameter_grid


def evaluate_dncnn_parameters(
        filters=None,
        depth=None,
        noise_std_train=None,
        bn=None,
        n_samples=None,
        exp_id='',
        **add_kwargs,
    ):
    if filters is None:
        filters = [64]
    if depth is None:
        depth = [20]
    if bn is None:
        bn = [False]
    if noise_std_train is None:
        noise_std_train = [(0, 55)]
    if n_samples is None:
        n_samples = [None]
    exp_id = f'dncnn{exp_id}'
    param_grid = {
        'filters': filters,
        'depth': depth,
        'noise_std_train': noise_std_train,
        'bn': bn,
        'n_samples': n_samples,
    }
    param_grid.update(**add_kwargs)
    metrics_names, results = train_eval_parameter_grid(
        exp_id,
        train_dncnn,
        evaluate_dncnn,
        parameter_grid=param_grid,
    )
    results_df = results_to_csv(
        metrics_names,
        results,
        output_path=f'{exp_id}.csv',
    )
    return results_df
