from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier
import numpy as np
from scipy.stats import ttest_ind

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'pareto'
n_runs = 5
n_gpus = 4
n_refines = [0, 1, 2, 7, 10, None]
base_params = dict(
    model_size='LARGE',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=220,
)
parameters = []
for i_run in range(n_runs):
    base_params.update(seed=i_run)
    for n_refine in n_refines:
        base_params.update(n_refine=n_refine)
        if n_refine != 0:
            parameters += [
                dict(**base_params),
            ]
        parameters += [
            dict(shine=True, refine=True, **base_params),
            dict(fpn=True, refine=True, **base_params),
        ]

res_all = train_eval_grid(
    job_name,
    train_classifier,
    evaluate_classifier,
    parameters,
    to_grid=False,
    timeout_train=20,
    n_gpus_train=n_gpus,
    timeout_eval=2,
    n_gpus_eval=n_gpus,
    project='shine',
    params_to_ignore=['n_epochs'],
    torch=True,
    no_force_32=True,
)

for n_refine in n_refines:
    perf_orig = [
        res for (res, params) in zip(res_all, parameters)
        if not params.get('shine', False) and not params.get('fpn', False) and params['n_refine'] == n_refine
    ]
    perf_shine = [
        res for (res, params) in zip(res_all, parameters)
        if params.get('shine', False) and params['n_refine'] == n_refine
    ]
    perf_fpn = [
        res for (res, params) in zip(res_all, parameters)
        if params.get('fpn', False) and params['n_refine'] == n_refine
    ]


    print(f'Descriptive stats for {n_refine}')
    print('Perf orig', np.mean(perf_orig), np.std(perf_orig))
    print('Perf shine', np.mean(perf_shine), np.std(perf_shine))
    print('Perf fpn', np.mean(perf_fpn), np.std(perf_fpn))
