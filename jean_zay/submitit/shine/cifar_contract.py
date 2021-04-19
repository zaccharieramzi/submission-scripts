from mdeq_lib.evaluate.cls_valid import evaluate_classifier

from jean_zay.submitit.general_submissions import eval_grid


job_name = 'shine_classifier_cifar_large_contract'
n_gpus = 1
base_params = dict(
    model_size='LARGE',
    dataset='cifar',
    n_gpus=n_gpus,
    check_contract=True,
    n_iter=500,
    seed=0,
)
parameters = []
parameters += [
    base_params,
    dict(shine=True, **base_params),
    dict(fpn=True, **base_params),
]

res_all = eval_grid(
    job_name,
    evaluate_classifier,
    parameters,
    to_grid=False,
    timeout=20,
    n_gpus=n_gpus,
    project='shine',
    params_to_ignore=['n_epochs'],
    torch=True,
    no_force_32=True,
)
