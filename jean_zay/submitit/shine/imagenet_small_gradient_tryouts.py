from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_imagenet_small_adam'
n_gpus = 4
base_params = dict(
    dataset='imagenet',
    n_gpus=n_gpus,
    n_epochs=100,
    seed=0,
    save_at=48,
)
parameters = [
    # dict(shine=True, model_size='SMALL_clip', restart_from=48, **base_params),
    # dict(shine=True, model_size='SMALL_adam', **base_params),
    dict(model_size='SMALL_adam', restart_from=48, fallback=True, **base_params),
]

train_eval_grid(
    job_name,
    train_classifier,
    evaluate_classifier,
    parameters,
    to_grid=False,
    timeout_train=100,
    n_gpus_train=n_gpus,
    timeout_eval=20,
    n_gpus_eval=n_gpus,
    project='shine',
    params_to_ignore=['n_epochs', 'restart_from', 'save_at'],
    torch=True,
)
