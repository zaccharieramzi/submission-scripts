from mdeq_lib.tests.modules.adj_broyden_convergence import save_results

from jean_zay.submitit.general_submissions import get_executor


job_name = 'adj_broyden_conv'
n_gpus = 1
base_params = dict(
    n_runs=100,
)
parameters = [
    dict(dataset='imagenet', model_size='SMALL', **base_params),
    dict(dataset='cifar', model_size='LARGE', **base_params),
]

executor = get_executor(job_name, timeout_hour=2, n_gpus=n_gpus, project='shine')
jobs = []
with executor.batch():
    for param in parameters:
        job = executor.submit(save_results, **param)
        jobs.append(job)
