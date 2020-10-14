from learning_wavelets.evaluate_tmp.learnlet_eval import evaluate_learnlet
from learning_wavelets.evaluate_tmp.results_to_csv import results_to_csv
from learning_wavelets.training.scripts.learnlet_subclassed_training import train_learnlet

from generic_dask import train_eval_parameter_grid, eval_parameter_grid


def evaluate_learnlet_parameters(
        denoising_activation=None,
        n_filters=None,
        undecimated=None,
        exact_reco=None,
        n_reweights=None,
        noise_std_train=None,
        n_samples=None,
        exp_id='',
        **add_params,
    ):
    if denoising_activation is None:
        denoising_activation = ['dynamic_soft_thresholding']
    if n_filters is None:
        n_filters = [256]
    if undecimated is None:
        undecimated = [True]
    if exact_reco is None:
        exact_reco = [True]
    if n_reweights is None:
        n_reweights = [1]
    if noise_std_train is None:
        noise_std_train = [(0, 55)]
    if n_samples is None:
        n_samples = [None]
    exp_id = f'learnlet{exp_id}'
    param_grid = {
        'denoising_activation': denoising_activation,
        'n_filters': n_filters,
        'undecimated': undecimated,
        'exact_reco': exact_reco,
        'n_reweights': n_reweights,
        'noise_std_train': noise_std_train,
        'n_samples': n_samples,
    }
    param_grid.update(**add_params)
    metrics_names, results = train_eval_parameter_grid(
        exp_id,
        train_learnlet,
        evaluate_learnlet,
        parameter_grid=param_grid,
    )
    results_df = results_to_csv(
        metrics_names,
        results,
        output_path=f'{exp_id}.csv',
    )
    return results_df


def evaluate_learnlet_runs(
        run_ids,
        denoising_activation=None,
        n_filters=None,
        undecimated=None,
        exact_reco=None,
        n_reweights=None,
        noise_std_train=None,
        n_samples=None,
        exp_id='',
        **add_kwargs,
    ):
    if denoising_activation is None:
        denoising_activation = ['dynamic_soft_thresholding']
    if n_filters is None:
        n_filters = [256]
    if undecimated is None:
        undecimated = [True]
    if exact_reco is None:
        exact_reco = [True]
    if n_reweights is None:
        n_reweights = [1]
    if noise_std_train is None:
        noise_std_train = [(0, 55)]
    if n_samples is None:
        n_samples = [None]
    exp_id = f'learnlet{exp_id}'
    param_grid = {
        'denoising_activation': denoising_activation,
        'n_filters': n_filters,
        'undecimated': undecimated,
        'exact_reco': exact_reco,
        'n_reweights': n_reweights,
        'noise_std_train': noise_std_train,
        'n_samples': n_samples,
    }
    param_grid.update(**add_kwargs)
    metrics_names, results = eval_parameter_grid(
        exp_id,
        evaluate_learnlet,
        parameter_grid=param_grid,
        run_ids=run_ids,
    )
    results_df = results_to_csv(
        metrics_names,
        results,
        output_path=f'{exp_id}.csv',
    )
    return results_df
