from sklearn.model_selection import ParameterGrid
import submitit


def get_executor(job_name, timeout_hour=60, n_gpus=1):
    executor = submitit.AutoExecutor(folder=job_name)
    if timeout_hour > 20:
        qos = 't4'
    elif timeout_hour > 2:
        qos = 't3'
    else:
        qos = 'dev'
    executor.update_parameters(
        slurm_job_name=job_name,
        slurm_time=f'{timeout_hour}:00:00',
        slurm_gres=f'gpu:{n_gpus}',
        slurm_additional_parameters={
            'ntasks': 1,
            'cpus-per-task':  10 * n_gpus,
            'account': 'hih@gpu',
            'qos': f'qos_gpu-{qos}',
            'distribution': 'block:block',
            'hint': 'nomultithread',
        },
        slurm_setup=[
            'cd $WORK/submission-scripts/jean_zay/env_configs',
            '. fastmri.sh',
        ],
    )
    return executor

def train_eval_grid(
        job_name,
        train_function,
        eval_function,
        parameter_grid,
        n_samples_eval=50,
        to_grid=True,
        timeout_train=60,
        n_gpus_train=4,
        timeout_eval=10,
        n_gpus_eval=4,
        **specific_eval_params,
    ):
    if to_grid:
        parameters = list(ParameterGrid(parameter_grid))
    else:
        parameters = parameter_grid
    executor = get_executor(job_name, timeout_hour=timeout_train, n_gpus=n_gpus_train)
    jobs = []
    with executor.batch():
        for param in parameters:
            job = executor.submit(train_function, **param)
            jobs.append(job)
    run_ids = [job.result() for job in jobs]
    print(run_ids)
    eval_grid(
        run_ids,
        job_name,
        eval_function,
        parameter_grid,
        n_samples=n_samples_eval,
        to_grid=to_grid,
        timeout=timeout_eval,
        n_gpus=n_gpus_eval,
        **specific_eval_params,
    )

def eval_grid(
        run_ids,
        job_name,
        eval_function,
        parameter_grid,
        n_samples=50,
        to_grid=True,
        timeout=10,
        n_gpus=4,
        **specific_eval_params,
    ):
    if to_grid:
        parameters = list(ParameterGrid(parameter_grid))
    else:
        parameters = parameter_grid
    executor = get_executor(job_name, timeout_hour=timeout, n_gpus=n_gpus)
    original_parameters = []
    for params in parameters:
        original_params = {}
        original_params['loss'] = params.pop('loss', 'mae')
        original_params['n_samples'] = params.pop('n_samples', None)
        original_parameters.append(original_params)
    jobs = []
    with executor.batch():
        for run_id, param in zip(run_ids, parameters):
            kwargs = param
            kwargs.update(**specific_eval_params)
            job = executor.submit(eval_function, run_id=run_id, n_samples=n_samples, **param)
            jobs.append(job)
    eval_results = []
    for param, original_param, job in zip(parameters, original_parameters, jobs):
        metrics_names, eval_res = job.result()
        param.update(original_param)
        print('Parameters', param)
        print(metrics_names)
        print(eval_res)
        eval_results.append(eval_res)
    return eval_results
