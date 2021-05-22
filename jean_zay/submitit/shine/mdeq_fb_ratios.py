import numpy as np
from mdeq_lib.debug.backward_forward_ratio import eval_ratio_fb_classifier

from jean_zay.submitit.general_submissions import get_executor


job_name = 'debug_shine_ratio_fb'
n_gpus = 1
base_params = dict(
    n_gpus=n_gpus,
    n_epochs=100,
    seed=42,
    restart_from=50,
    n_samples=200,
)
parameters = [
    dict(dataset='cifar', model_size='LARGE', **base_params),
    dict(dataset='imagenet', model_size='SMALL', **base_params),
]

executor = get_executor(job_name, timeout_hour=2, n_gpus=n_gpus, project='shine')
jobs = []
with executor.batch():
    for param in parameters:
        job = executor.submit(eval_ratio_fb_classifier, **param)
        jobs.append(job)
results = [job.result() for job in jobs]
res_cifar = results[0]
res_imagenet = results[1]

print('detail cifar', res_cifar)
print('detail imagenet', res_imagenet)


print('Res CIFAR', np.median(res_cifar[1:]), np.std(res_cifar[1:]))
print('Res ImageNet', np.median(res_imagenet[1:]), np.std(res_imagenet[1:]))
