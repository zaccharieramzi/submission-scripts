from tf_soft_thresholding.training.train_denoisers import train_dncnn as train
from tf_soft_thresholding.evaluate.evaluate_denoisers import evaluate_dncnn as evaluate

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid


job_name = 'dncnn_sota'
n_epochs = 100
to_grey = True
patch_size = 256
patch_size_eval = None
batch_size = 8
n_steps_per_epoch = 3000
batch_size_eval = 1
noise_config = dict(
    noise_input=True,
    noise_power_spec=55/255,
    noise_range_type='uniform',
)
noise_config_eval = dict(
    noise_input=True,
    fixed_noise=True,
    noise_power_spec=25/255,
)
n_samples_eval = 68
n_filters = 64
n_convs = 20
models = {
    'dncnn-relu': {},
    'dcnnn-relu-bn': {'bn': True},
}
parameter_grid = [
    dict(
        model_config=dict(n_filters=n_filters, n_convs=n_convs, **model_config),
        model_name=model_name,
        n_epochs=n_epochs,
        to_grey=to_grey,
        patch_size=patch_size,
        batch_size=batch_size,
        n_steps_per_epoch=n_steps_per_epoch,
        noise_config=noise_config,
        lr=1e-4,
    )
    for model_name, model_config in models.items()
]

eval_results, run_ids = train_eval_grid(
# run_ids = ['dncnn-relu_1606927954', 'dcnnn-relu-bn_1606927955']
# eval_results = eval_grid(
    job_name,
    train,
    evaluate,
    parameter_grid,
    # run_ids=run_ids,
    n_samples_eval=n_samples_eval,
    timeout_train=20,
    n_gpus_train=4,
    timeout_eval=4,
    n_gpus_eval=1,
    # n_samples=n_samples_eval,
    # timeout=10,
    # n_gpus=1,
    to_grid=False,
    patch_size=patch_size_eval,  # just for eval
    batch_size=batch_size_eval,  # just for eval
    noise_config=noise_config_eval,  # just for eval
    mode='bsd68',  # just for eval
    project='soft_thresholding',
    # return_run_ids=False,
)

print(eval_results)
