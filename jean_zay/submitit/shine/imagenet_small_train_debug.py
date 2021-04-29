from mdeq_lib.debug.cls_train_evolve import train_classifier

from jean_zay.submitit.general_submissions import get_executor


job_name = 'debug_shine'
n_gpus = 1
base_params = dict(
    model_size='SMALL',
    dataset='imagenet',
    n_gpus=n_gpus,
    n_epochs=100,
    seed=42,
    restart_from=48,
    gradient_correl=False,
    gradient_ratio=False,
    n_iter=200,
)
parameters = [
    # base_params,
    dict(shine=True, **base_params),
    # dict(fpn=True, **base_params),
]

executor = get_executor(job_name, timeout_hour=2, n_gpus=n_gpus, project='shine')
jobs = []
with executor.batch():
    for param in parameters:
        job = executor.submit(train_classifier, **param)
        jobs.append(job)
[job.result() for job in jobs]
