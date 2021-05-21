from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier
import numpy as np
from scipy.stats import ttest_ind

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_cifar_small_adjbroyden'
n_runs = 5
n_gpus = 4
base_params = dict(
    model_size='TINY',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=80,
    adjoint_broyden=True,
    save_at=14,
    restart_from=14,
)
parameters = []
for i_run in range(n_runs):
    base_params.update(seed=i_run)
    parameters += [
        # base_params,
        dict(shine=True, **base_params),
        dict(shine=True, opa=True, **base_params),
        # dict(fpn=True, **base_params),
    ]

res_all = train_eval_grid(
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
    params_to_ignore=['n_epochs', 'save_at', 'restart_from'],
    torch=True,
    no_force_32=True,
)

perf_shine_adj_br = [res for (res, params) in zip(res_all, parameters) if not params.get('opa', False)]
perf_shine_opa = [res for (res, params) in zip(res_all, parameters) if params.get('opa', False)]

print('Descriptive stats')
print('Perf shine adj broyden', np.mean(perf_shine_adj_br), np.std(perf_shine_adj_br))
print('Perf shine opa', np.mean(perf_shine_opa), np.std(perf_shine_opa))
