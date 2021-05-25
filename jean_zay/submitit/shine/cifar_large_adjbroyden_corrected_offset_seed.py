from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from mdeq_lib.training.cls_train import train_classifier
import numpy as np
from scipy.stats import ttest_ind

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_cifar_large_adjbroyden_corrected'
n_gpus = 4
n_runs = 5
base_params = dict(
    model_size='LARGE',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=220,
    save_at=50,
    restart_from=50,
    adjoint_broyden=True,
    shine=True,
    fallback=True,
)
parameters = []
for i_run in range(n_runs):
    parameters += [
        # dict(seed=i_run, **base_params),
        dict(seed=i_run+10, opa=True, **base_params),
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
    params_to_ignore=['n_epochs', 'save_at', 'restart_from'],
    torch=True,
    no_force_32=True,
)


perf_shine_opa = [res for (res, params) in zip(res_all, parameters) if params.get('opa', False)]

print('Descriptive stats')
print('Perf shine opa', np.mean(perf_shine_opa), np.std(perf_shine_opa))
