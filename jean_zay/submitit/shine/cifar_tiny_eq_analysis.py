from mdeq_lib.debug.eq_init_analysis import analyze_equilibrium_initialization

from jean_zay.submitit.general_submissions import get_executor, ParameterGrid


job_name = 'shine_eq_analy_cifar_tiny'
n_gpus = 2

executor = get_executor(job_name, n_gpus, timeout_hour=2, project='shine', no_force_32=True)

params = dict(
    model_size='TINY',
    dataset='cifar',
    n_samples_train=[64*n_gpus*5, 64*n_gpus*10, 50_000],
    n_images=100,
    checkpoint=[14, 40, 60, 70],
    on_cpu=False,
    n_gpus=n_gpus,
)
params = list(ParameterGrid(params))

jobs = []
with executor.batch():
    for param in params:
        job = executor.submit(analyze_equilibrium_initialization, **param)
        jobs.append(job)
df_res_list = [job.result() for job in jobs]
