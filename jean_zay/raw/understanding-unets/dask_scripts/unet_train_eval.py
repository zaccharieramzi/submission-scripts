from learning_wavelets.evaluate_tmp.unet_eval import evaluate_unet
from learning_wavelets.evaluate_tmp.results_to_csv import results_to_csv
from learning_wavelets.training.scripts.unet_training import train_unet

from generic_dask import train_eval_parameter_grid, eval_parameter_grid


def evaluate_unet_parameters(
        base_n_filters=None,
        noise_std_train=None,
        n_samples=None,
        exp_id='',
        **add_kwargs,
    ):
    if base_n_filters is None:
        base_n_filters = [64]
    if noise_std_train is None:
        noise_std_train = [(0, 55)]
    if n_samples is None:
        n_samples = [None]
    exp_id = f'unet{exp_id}'
    param_grid = {
        'base_n_filters': base_n_filters,
        'noise_std_train': noise_std_train,
        'n_samples': n_samples,
    }
    param_grid.update(**add_kwargs)
    metrics_names, results = train_eval_parameter_grid(
        exp_id,
        train_unet,
        evaluate_unet,
        parameter_grid=param_grid,
    )
    results_df = results_to_csv(
        metrics_names,
        results,
        output_path=f'{exp_id}.csv',
    )
    return results_df

def evaluate_unet_runs(
        run_ids,
        base_n_filters=None,
        noise_std_train=None,
        n_samples=None,
        exp_id='',
        **add_kwargs,
    ):
    if base_n_filters is None:
        base_n_filters = [64]
    if noise_std_train is None:
        noise_std_train = [(0, 55)]
    if n_samples is None:
        n_samples = [None]
    exp_id = f'unet{exp_id}'
    param_grid = {
        'base_n_filters': base_n_filters,
        'noise_std_train': noise_std_train,
        'n_samples': n_samples,
    }
    param_grid.update(**add_kwargs)
    metrics_names, results = eval_parameter_grid(
        exp_id,
        evaluate_unet,
        parameter_grid=param_grid,
        run_ids=run_ids,
    )
    results_df = results_to_csv(
        metrics_names,
        results,
        output_path=f'{exp_id}.csv',
    )
    return results_df
