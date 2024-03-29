from mdeq_lib.training.cls_train import train_classifier

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_cifar_tiny_pretrain'
n_gpus = 2
params = dict(
    model_size='TINY',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=80,
    save_at=[14, 40, 60, 70],
    seed=0,
)

def _dummy_eval(**kwargs):
    pass

train_eval_grid(
    job_name,
    train_classifier,
    _dummy_eval,
    [params],
    to_grid=False,
    timeout_train=20,
    n_gpus_train=n_gpus,
    timeout_eval=0,
    n_gpus_eval=0,
    project='shine',
    params_to_ignore=['n_epochs', 'save_at', 'restart_from'],
    torch=True,
)
