import pandas as pd
from tf_soft_thresholding.training.train_denoisers import train_dncnn as train
from tf_soft_thresholding.evaluate.evaluate_denoisers import evaluate_dncnn as evaluate

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'st_denoising'
n_epochs = 1
to_grey = True
patch_size = 64
patch_size_eval = None
batch_size = 32
batch_size_eval = 1
noise_levels = [0, 5, 15, 25, 30, 45, 50]
noise_config = dict(
    noise_input=True,
    noise_power_spec=noise_levels[-1]/255,
)
noise_config_eval = dict(
    noise_input=True,
    fixed_noise=True,
    noise_power_spec=noise_levels[-1]/255,
)
n_samples_eval = 20
models = {
    'dncnn-relu': {},
    'dncnn-prelu': {'activation': 'prelu'},
    'dncnn-st': {'activation': 'soft_thresh', 'st_kwargs': {'threshold': 0.2}, 'res': False},
    'dncnn-anti-st': {'activation': 'soft_thresh', 'st_kwargs': {'threshold': 0.2, 'anti_pattern': True}},
    'dncnn-sig': {'activation': 'sigmoid'},
    'dncnn-nc-relu': {'naive_noise_conditioning': True, },
    'dncnn-nc-prelu': {'naive_noise_conditioning': True, 'activation': 'prelu'},
    'dncnn-nc-st': {'naive_noise_conditioning': True, 'activation': 'soft_thresh', 'st_kwargs': {'threshold': 0.2}, 'res': False},
    'dncnn-nc-anti-st': {'naive_noise_conditioning': True, 'activation': 'soft_thresh', 'st_kwargs': {'threshold': 0.2, 'anti_pattern': True}},
    'dncnn-nc-sig': {'activation': 'sigmoid', 'naive_noise_conditioning': True},
    'dncnn-dst': {'res': False, 'activation': 'dynamic_soft_thresh', 'st_kwargs': {'alpha_init': 0.2}},
    'dncnn-dst-train': {'res': False, 'activation': 'dynamic_soft_thresh', 'st_kwargs': {'alpha_init': 0.2, 'trainable': True}},
    'dncnn-dst-train-per-fiter': {'res': False, 'activation': 'dynamic_soft_thresh', 'st_kwargs': {'alpha_init': 0.2, 'trainable': True, 'per_filter': True}},
    'dncnn-dst-train-per-filter-w-noise-est': {'res': False, 'activation': 'dynamic_soft_thresh', 'st_kwargs': {'alpha_init': 0.2, 'trainable': True, 'per_filter': True, 'noise_estimation': True}},
    'dncnn-dst-train-w-noise-est': {'res': False, 'activation': 'dynamic_soft_thresh', 'st_kwargs': {'alpha_init': 0.2, 'trainable': True, 'noise_estimation': True}},
    'dncnn-anti-dst': {'activation': 'dynamic_soft_thresh', 'st_kwargs': {'anti_pattern': True, 'alpha_init': 0.2}},
    'dncnn-anti-dst-train': {'activation': 'dynamic_soft_thresh', 'st_kwargs': {'anti_pattern': True, 'alpha_init': 0.2, 'trainable': True}},
    'dncnn-anti-dst-train-per-fiter': {'activation': 'dynamic_soft_thresh', 'st_kwargs': {'anti_pattern': True, 'alpha_init': 0.2, 'trainable': True, 'per_filter': True}},
    'dncnn-anti-dst-train-per-filter-w-noise-est': {'activation': 'dynamic_soft_thresh', 'st_kwargs': {'anti_pattern': True, 'alpha_init': 0.2, 'trainable': True, 'per_filter': True, 'noise_estimation': True}},
    'dncnn-anti-dst-train-w-noise-est': {'activation': 'dynamic_soft_thresh', 'st_kwargs': {'anti_pattern': True, 'alpha_init': 0.2, 'trainable': True, 'noise_estimation': True}},
}
parameter_grid = [
    dict(
        model_config=dict(**model_config),
        model_name=model_name,
        n_epochs=n_epochs,
        to_grey=to_grey,
        patch_size=patch_size,
        batch_size=batch_size,
        noise_config=noise_config,
    )
    for model_name, model_config in models.items()
]

eval_results_50, run_ids = train_eval_grid(
# eval_results = eval_grid(
    job_name,
    train,
    evaluate,
    parameter_grid,
    n_samples_eval=n_samples_eval,
    timeout_train=20,
    n_gpus_train=1,
    timeout_eval=4,
    n_gpus_eval=1,
    # n_samples=200,
    # timeout=10,
    # n_gpus=1,
    to_grid=False,
    patch_size=patch_size_eval,  # just for eval
    batch_size=batch_size_eval,  # just for eval
    noise_config=noise_config_eval,  # just for eval
    project='soft_thresholding',
    return_run_ids=True,
)

eval_res = []
for noise_level in noise_levels[:-1]:
    noise_config_eval = dict(
        noise_input=True,
        fixed_noise=True,
        noise_power_spec=noise_level/255,
    )
    eval_results = eval_grid(
        job_name,
        # train,
        evaluate,
        parameter_grid,
        run_ids=run_ids,
        # n_samples_eval=20,
        # timeout_train=20,
        # n_gpus_train=1,
        # timeout_eval=4,
        # n_gpus_eval=1,
        n_samples=n_samples_eval,
        timeout=4,
        n_gpus=1,
        to_grid=False,
        patch_size=patch_size_eval,  # just for eval
        batch_size=batch_size_eval,  # just for eval
        noise_config=noise_config_eval,  # just for eval
        project='soft_thresholding',
    )
    eval_res.append(eval_results)

eval_res.append(eval_results_50)
data_for_df = []
for eval_results, noise_level in zip(eval_res, noise_levels):
    for metrics, model_name in zip(eval_results, models.keys()):
        data_for_df.append([
            model_name,
            noise_level,
            metrics[0],
            metrics[1],
        ])
df_results = pd.DataFrame(data_for_df, columns='model_name noise_level psnr ssim'.split())

print(df_results)
outputs_file = 'denoising_results_dncnn.csv'
df_results.to_csv(outputs_file)
