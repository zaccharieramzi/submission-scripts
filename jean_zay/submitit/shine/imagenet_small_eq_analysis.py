from mdeq_lib.debug.eq_init_analysis import analyze_equilibrium_initialization

from jean_zay.submitit.general_submissions import get_executor, ParameterGrid


job_name = 'shine_eq_analy_imagenet_small'
n_gpus = 4

executor = get_executor(job_name, n_gpus, timeout_hour=20, project='shine', no_force_32=False)

params = dict(
    model_size='SMALL',
    dataset='imagenet',
    n_samples_train=[32*n_gpus*500, 32*n_gpus*1000, 1_281_167],
    n_images=100,
    checkpoint=[50, 100, 150, 200],
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
