from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier
import numpy as np
from scipy.stats import ttest_ind

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_cifar_small'
n_runs = 5
n_gpus = 4
base_params = dict(
    model_size='TINY',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=80,
)

perf_orig = []
perf_shine = []
perf_fpn = []
for i_run in range(n_runs):
    base_params.update(seed=i_run)
    parameters = [
        base_params,
        dict(shine=True, **base_params),
        dict(fpn=True, **base_params),
    ]

    res_orig, res_shine, res_fpn = train_eval_grid(
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
    perf_orig.append(res_orig)
    perf_shine.append(res_shine)
    perf_fpn.append(res_fpn)

print('Perf orig', perf_orig)
print('Perf shine', perf_shine)
print('Perf fpn', perf_fpn)

print('Stats test orig vs shine', ttest_ind(perf_orig, perf_shine))
print('Stats test orig vs fpn', ttest_ind(perf_orig, perf_fpn))

print('Descriptive stats')
print('Perf orig', np.mean(perf_orig), np.std(perf_orig))
print('Perf shine', np.mean(perf_shine), np.std(perf_shine))
print('Perf fpn', np.mean(perf_fpn), np.std(perf_fpn))
