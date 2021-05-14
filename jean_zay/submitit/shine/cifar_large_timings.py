from mdeq_lib.debug.cls_grad_time import train_classifier

from jean_zay.submitit.general_submissions import get_executor


job_name = 'debug_shine'
n_gpus = 1
base_params = dict(
    model_size='LARGE',
    dataset='cifar',
    n_gpus=n_gpus,
    n_epochs=220,
    seed=0,
    gradient_correl=False,
    gradient_ratio=False,
    compute_partial=True,
    f_thres_range=range(18, 19),
    n_samples=300,
)
# n_refines = [0, 1, 2, 5, 7, 10, None]
n_refines = [0]

parameters = []
for n_refine in n_refines:
    refine_active = n_refine > 0 or n_refine is None
    if refine_active:
        base_params.update(n_refine=n_refine)
    if n_refine != 0:
        parameters += [
            dict(**base_params),
        ]
    parameters += [
        dict(shine=True, refine=refine_active, **base_params),
        dict(fpn=True, refine=refine_active, **base_params),
    ]


executor = get_executor(job_name, timeout_hour=2, n_gpus=n_gpus, project='shine')
jobs = []
with executor.batch():
    for param in parameters:
        job = executor.submit(train_classifier, **param)
        jobs.append(job)
results = [job.result() for job in jobs]

for param, res in zip(parameters, results):
    if param.get('shine', False):
        name_method = 'shine'
    elif param.get('fpn', False):
        name_method = 'fpn'
    else:
        name_method = 'original'
    print(name_method, param.get('n_refine', 0), res)
