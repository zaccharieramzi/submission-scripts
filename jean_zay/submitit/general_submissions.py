from sklearn.model_selection import ParameterGrid
import submitit


def get_executor(job_name, timeout_hour=60, n_gpus=1, project='fastmri'):
    executor = submitit.AutoExecutor(folder=job_name)
    if timeout_hour > 20:
        qos = 't4'
    elif timeout_hour > 2:
        qos = 't3'
    else:
        qos = 'dev'
    executor.update_parameters(
        timeout_min=60,
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
            '#SBATCH -C v100-32g',
            'cd $WORK/submission-scripts/jean_zay/env_configs',
            f'. {project}.sh',
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
        project='fastmri',
        return_run_ids=False,
        params_to_ignore=None,
        checkpoints_train=0,
        **specific_eval_params,
    ):
    if to_grid:
        parameters = list(ParameterGrid(parameter_grid))
    else:
        parameters = parameter_grid
    executor = get_executor(
        job_name,
        timeout_hour=timeout_train,
        n_gpus=n_gpus_train,
        project=project,
    )
    jobs = []
    with executor.batch():
        for param in parameters:
            if checkpoints_train:
                new_param = dict(**param)
                new_param['n_epochs'] //= (checkpoints_train+1)
                new_param['save_state'] = True
            else:
                new_param = param
            job = executor.submit(train_function, **new_param)
            jobs.append(job)
    run_ids = [job.result() for job in jobs]
    print(run_ids)
    # extra-loop for checkpointing
    if checkpoints_train:
        for i_checkpoint in range(1, checkpoints_train+1):
            with executor.batch():
                for orig_run_id, param in zip(run_ids, parameters):
                    new_param = dict(**param)
                    new_param['n_epochs'] //= (checkpoints_train+1)
                    new_param['save_state'] = i_checkpoint < checkpoints_train
                    new_param['checkpoint_epoch'] = new_param['n_epochs'] * i_checkpoint
                    new_param['original_run_id'] = orig_run_id
                    job = executor.submit(train_function, **new_param)
                    jobs.append(job)
            run_ids = [job.result() for job in jobs]
    if return_run_ids:
        return eval_grid(
            job_name,
            eval_function,
            parameter_grid,
            run_ids=run_ids,
            n_samples=n_samples_eval,
            to_grid=to_grid,
            timeout=timeout_eval,
            n_gpus=n_gpus_eval,
            project=project,
            params_to_ignore=params_to_ignore,
            **specific_eval_params,
        ), run_ids
    else:
        return eval_grid(
            job_name,
            eval_function,
            parameter_grid,
            run_ids=run_ids,
            n_samples=n_samples_eval,
            to_grid=to_grid,
            timeout=timeout_eval,
            n_gpus=n_gpus_eval,
            project=project,
            params_to_ignore=params_to_ignore,
            **specific_eval_params,
        )

def eval_grid(
        job_name,
        eval_function,
        parameter_grid,
        run_ids=None,
        n_samples=50,
        to_grid=True,
        timeout=10,
        n_gpus=4,
        project='fastmri',
        params_to_ignore=None,
        **specific_eval_params,
    ):
    if to_grid:
        parameters = list(ParameterGrid(parameter_grid))
    else:
        parameters = parameter_grid
    executor = get_executor(
        job_name,
        timeout_hour=timeout,
        n_gpus=n_gpus,
        project=project,
    )
    original_parameters = []
    if params_to_ignore is None:
        params_to_ignore = []
    params_to_ignore += [
        'loss',
        'n_samples',
        'run_id',
        'n_steps_per_epoch',
        'model_size',
        'model_name',
        'lr',
    ]
    for params in parameters:
        original_params = {}
        params_keys = list(params.keys())
        for param in params_keys:
            if param in params_to_ignore:
                original_params[param] = params.pop(param, None)
        original_parameters.append(original_params)
    jobs = []
    with executor.batch():
        if run_ids is None:
            run_ids = [None]*len(parameters)
        for run_id, param in zip(run_ids, parameters):
            kwargs = param
            kwargs.update(**specific_eval_params)
            if run_id is not None:
                kwargs.update(run_id=run_id)
            job = executor.submit(eval_function, n_samples=n_samples, **kwargs)
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

def infer_grid(
        job_name,
        infer_function,
        parameter_grid,
        run_ids=None,
        to_grid=True,
        timeout=10,
        n_gpus=4,
        project='fastmri',
        params_to_ignore=None,
        **specific_infer_params,
    ):
    if to_grid:
        parameters = list(ParameterGrid(parameter_grid))
    else:
        parameters = parameter_grid
    executor = get_executor(
        job_name,
        timeout_hour=timeout,
        n_gpus=n_gpus,
        project=project,
    )
    original_parameters = []
    if params_to_ignore is None:
        params_to_ignore = []
    params_to_ignore += [
        'loss',
        'n_samples',
        'run_id',
        'n_steps_per_epoch',
        'model_size',
        'model_name',
        'lr',
    ]
    for params in parameters:
        original_params = {}
        params_keys = list(params.keys())
        for param in params_keys:
            if param in params_to_ignore:
                original_params[param] = params.pop(param, None)
        original_parameters.append(original_params)
    jobs = []
    with executor.batch():
        if run_ids is None:
            run_ids = [None]*len(parameters)
        for run_id, param in zip(run_ids, parameters):
            kwargs = param
            kwargs.update(**specific_infer_params)
            if run_id is not None:
                kwargs.update(run_id=run_id)
            job = executor.submit(infer_function, exp_id=job_name, **kwargs)
            jobs.append(job)
    for job in jobs:
        job.result()
