from mdeq_lib.training.cls_train import train_classifier

from jean_zay.submitit.general_submissions import get_executor


job_name = 'shine_classifier_cifar_small'
n_gpus = 4
executor = get_executor(job_name, timeout_hour=20, n_gpus=n_gpus, project='shine')

with executor.batch():
    for shine in [True, False]:
        executor.submit(
            train_classifier,
            model_size='TINY',
            dataset='cifar',
            n_gpus=n_gpus,
            shine=shine,
            n_epochs=50,
        )
