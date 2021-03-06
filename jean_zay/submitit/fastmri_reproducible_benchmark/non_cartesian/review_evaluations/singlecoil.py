from fastmri_recon.evaluate.scripts.nc_eval import evaluate_dcomp
from fastmri_recon.evaluate.scripts.nc_eval import evaluate_nc_multinet as eval_fun

from jean_zay.submitit.general_submissions import get_executor, eval_grid, ParameterGrid


n_samples = 100
grid_params = dict(
    n_samples=n_samples,
    n_gpus=1,
    wait=False,
    no_force_32=True,
    project='fastmri4',
)

base_job_name = 'ncsc_eval'

common_base_params = dict(
    contrast=['CORPD_FBK', 'CORPDFS_FBK'],
    af=[4, 8],
    acq_type=['radial', 'spiral'],
    n_epochs=[100],
)

def deduplicate_run_ids(base_run_ids):
    new_run_ids = [run_id for quadruple in zip(*([base_run_ids]*4)) for run_id in quadruple]
    return new_run_ids

def extend_params(params):
    extended_params = list(ParameterGrid(params))
    return extended_params

#### PDNet
pdnet_params = dict(
    model=['pdnet'],
    **common_base_params,
)

pdnet_params = [
  dict(dcomp=[True], normalize_image=[False], **pdnet_params),
  dict(dcomp=[False], normalize_image=[True], **pdnet_params),
]
run_ids = [
    'ncpdnet_singlecoil___radial_compound_mssim_dcomp_1610872636',
    'ncpdnet_singlecoil___spiral_compound_mssim_dcomp_1610911070',
    'ncpdnet_singlecoil___radial_compound_mssim_1610720754',
    'ncpdnet_singlecoil___spiral_compound_mssim_1610720754',
]

pdnet_jobs = eval_grid(
    base_job_name + '_pdnet',
    eval_fun,
    pdnet_params,
    run_ids=deduplicate_run_ids(run_ids),
    **grid_params,
)

#### PDNet gridded
pdnet_gridded_params = dict(
    model=['pdnet-gridded'],
    **common_base_params,
)

run_ids = [
    'pdnet_gridded_singlecoil___radial_compound_mssim_1611913690',
    'pdnet_gridded_singlecoil___spiral_compound_mssim_1611913692',
]

pdnet_gridded_jobs = eval_grid(
    base_job_name + '_pdnet_gridded',
    eval_fun,
    pdnet_gridded_params,
    run_ids=deduplicate_run_ids(run_ids),
    **grid_params,
)


#### Unet
unet_params = dict(
    model=['unet'],
    dcomp=[True],
    **common_base_params,
)

run_ids = [
    'unet_singlecoil___radial_compound_mssim_dcomp_1610911070',
    'unet_singlecoil___spiral_compound_mssim_dcomp_1610911070',
]

unet_jobs = eval_grid(
    base_job_name + '_unet',
    eval_fun,
    unet_params,
    run_ids=deduplicate_run_ids(run_ids),
    **grid_params,
)


#### Adj + DC
executor = get_executor(base_job_name + 'adjoint_dc', timeout_hour=10, n_gpus=1, project='fastmri4')
adj_dc_jobs = []
with executor.batch():
    for acq_type in common_base_params['acq_type']:
        for contrast in common_base_params['contrast']:
            job = executor.submit(
                evaluate_dcomp,
                acq_type=acq_type,
                af=common_base_params['af'][0],
                n_samples=n_samples,
                contrast=contrast,
            )
            adj_dc_jobs.append(job)
job_counter = 0
for acq_type in common_base_params['acq_type']:
    for contrast in common_base_params['contrast']:
        metrics_names, eval_res = adj_dc_jobs[job_counter].result()
        print('Parameters for Adj+DC', acq_type, contrast)
        print(eval_res)
        job_counter += 1

### Results presentation
### Nets
jobs = pdnet_jobs + pdnet_gridded_jobs + unet_jobs
params = extend_params(pdnet_params) + extend_params(pdnet_gridded_params) + extend_params(unet_params)
for param, job in zip(params, jobs):
    metrics_names, eval_res = job.result()
    print('Parameters', param)
    print(eval_res)
