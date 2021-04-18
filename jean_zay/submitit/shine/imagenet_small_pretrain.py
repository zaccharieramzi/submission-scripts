from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier
import numpy as np
from scipy.stats import ttest_ind

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_imagenet_small'
n_gpus = 4
n_runs = 1
base_params = dict(
    model_size='SMALL',
    dataset='imagenet',
    n_gpus=n_gpus,
    n_epochs=100,
    save_at=48,
)
parameters = []
for i_run in range(n_runs):
    parameters += [
        dict(seed=i_run, **base_params),
        # dict(seed=i_run, shine=True, **base_params),
        # dict(seed=i_run, fpn=True, **base_params),
    ]

res_all = train_eval_grid(
    job_name,
    train_classifier,
    evaluate_classifier,
    parameters,
    to_grid=False,
    timeout_train=10,
    n_gpus_train=n_gpus,
    timeout_eval=2,
    n_gpus_eval=n_gpus,
    project='shine',
    params_to_ignore=['n_epochs', 'save_at'],
    torch=True,
)


perf_orig = [res for (res, params) in zip(res_all, parameters) if not params.get('shine', False) and not params.get('fpn', False)]
perf_shine = [res for (res, params) in zip(res_all, parameters) if params.get('shine', False)]
perf_fpn = [res for (res, params) in zip(res_all, parameters) if params.get('fpn', False)]


print('Perf orig', perf_orig)
print('Perf shine', perf_shine)
print('Perf fpn', perf_fpn)

print('Stats test orig vs shine', ttest_ind(perf_orig, perf_shine))
print('Stats test orig vs fpn', ttest_ind(perf_orig, perf_fpn))

print('Descriptive stats')
print('Perf orig', np.mean(perf_orig), np.std(perf_orig))
print('Perf shine', np.mean(perf_shine), np.std(perf_shine))
print('Perf fpn', np.mean(perf_fpn), np.std(perf_fpn))
