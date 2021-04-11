from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_cifar_small_gradient_correl'
n_runs = 1
n_gpus = 4
base_params = dict(
    model_size='TINY',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=80,
    gradient_correl=True,
)
parameters = []
for i_run in range(n_runs):
    base_params.update(seed=5+i_run)
    parameters += [
        dict(shine=True, **base_params),
        dict(fpn=True, **base_params),
    ]

train_eval_grid(
    job_name,
    train_classifier,
    evaluate_classifier,
    parameters,
    to_grid=False,
    timeout_train=2,
    n_gpus_train=n_gpus,
    timeout_eval=1,
    n_gpus_eval=n_gpus,
    project='shine',
    params_to_ignore=['n_epochs'],
    torch=True,
    no_force_32=True,
)
