from mdeq_lib.evaluate.cls_valid import evaluate_classifier
from scipy.stats import ttest_ind

from jean_zay.submitit.general_submissions import eval_grid


job_name = 'debug_shine_eval'
n_gpus = 4
base_params = dict(
    model_size='TINY',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=80,
)
parameters = [
    dict(shine=True, **base_params),
    dict(fpn=True, **base_params),
]

res_shine, res_fpn = eval_grid(
    job_name,
    evaluate_classifier,
    parameters,
    to_grid=False,
    timeout=1,
    n_gpus=n_gpus,
    project='shine',
    params_to_ignore=['n_epochs'],
    torch=True,
    no_force_32=True,
)
