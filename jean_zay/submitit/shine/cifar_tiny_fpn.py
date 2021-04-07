from mdeq_lib.training.cls_train import train_classifier

from jean_zay.submitit.general_submissions import get_executor


job_name = 'shine_classifier_cifar_small_fpn'
n_gpus = 4
executor = get_executor(
    job_name,
    timeout_hour=1,
    n_gpus=n_gpus,
    project='shine',
    torch=True,
    no_force_32=True,
)

executor.submit(
    train_classifier,
    model_size='TINY',
    dataset='cifar',
    n_gpus=n_gpus,
    shine=False,
    fpn=True,
    n_epochs=25,
)
