from mdeq_lib.debug.eq_init_analysis import analyze_equilibrium_initialization

from jean_zay.submitit.general_submissions import get_executor, ParameterGrid


job_name = 'shine_eq_analy_cifar_large'
n_gpus = 1

executor = get_executor(
    job_name,
    n_gpus=n_gpus,
    timeout_hour=3,
    project='shine',
    no_force_32=False,
    force_partition='gpu_p2',
)

params = dict(
    model_size=['LARGE_reg'],
    dataset=['cifar'],
    n_samples_train=[32*n_gpus*5, 32*n_gpus*10, 50_000],
    n_images=[100],
    checkpoint=[50, 100, 150, 200],
    on_cpu=[False],
    n_gpus=[n_gpus],
    dropout_eval=[True, False],
)
params = list(ParameterGrid(params))

jobs = []
with executor.batch():
    for param in params:
        job = executor.submit(analyze_equilibrium_initialization, **param)
        jobs.append(job)
df_res_list = [job.result() for job in jobs]
