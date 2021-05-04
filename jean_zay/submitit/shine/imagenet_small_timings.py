from mdeq_lib.debug.cls_grad_time import train_classifier

from jean_zay.submitit.general_submissions import get_executor


job_name = 'debug_shine'
n_gpus = 1
base_params = dict(
    dataset='imagenet',
    n_gpus=n_gpus,
    n_epochs=100,
    seed=42,
    restart_from='50',
    gradient_correl=False,
    gradient_ratio=False,
    compute_partial=True,
    f_thres_range=range(26, 27),
    n_samples=300,
)
parameters = [
    # dict(model_size='SMALL', shine=True, **base_params),
    # dict(model_size='SMALL', fpn=True, **base_params),
    # dict(model_size='SMALL', shine=True, fallback=True, **base_params),
    dict(model_size='SMALL_refine', refine=True, shine=True, fallback=True, **base_params),
    dict(model_size='SMALL_refine', refine=True, fpn=True, **base_params),
]

executor = get_executor(job_name, timeout_hour=2, n_gpus=n_gpus, project='shine')
jobs = []
with executor.batch():
    for param in parameters:
        job = executor.submit(train_classifier, **param)
        jobs.append(job)
[job.result() for job in jobs]
