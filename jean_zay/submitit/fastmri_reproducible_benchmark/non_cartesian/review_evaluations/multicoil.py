from fastmri_recon.evaluate.scripts.nc_eval import evaluate_dcomp
from fastmri_recon.evaluate.scripts.nc_eval import evaluate_nc_multinet as eval_fun

from jean_zay.submitit.general_submissions import get_executor, eval_grid

n_samples = 100
grid_params = dict(
    n_samples=n_samples,
    n_gpus=3,
    wait=False,
    no_force_32=False,
    project='fastmri4',
)

base_job_name = 'ncmc_eval'
contrasts_knee = ['CORPD_FBK', 'CORPDFS_FBK']
contrasts_brain = ['AXT1', 'AXT2', 'AXT1POST', 'AXT1PRE', 'AXFLAIR']
common_base_params = dict(
    acq_type=['radial', 'spiral'],
    n_epochs=[100],
    multicoil=[True],
    refine_smaps=[True],
    dcomp=[True],
)

def from_base_params_to_multiple_params(base_params):
    multiple_params = [
        # classical eval
        dict(contrast=contrasts_knee, af=[4], **base_params),
        # af 8 eval
        dict(contrast=contrasts_knee, af=[8], **base_params),
        # reversed eval
        dict(contrast=contrasts_knee, af=[4], **base_params),
        # brain eval
        dict(contrast=contrasts_brain, af=[4], brain=[True], **base_params)
    ]
    return multiple_params

def duplicate_run_ids(base_run_ids):
    new_run_ids = [run_id for pair in zip(base_run_ids, base_run_ids) for run_id in pair]
    return new_run_ids

def deduplicate_run_ids(base_run_ids):
    new_run_ids = [run_id for quintuple in zip(*([base_run_ids]*5)) for run_id in quintuple]
    return new_run_ids

def extend_params(params):
    extended_params = list(ParameterGrid(params))
    return extended_params

#### PDNet
pdnet_params = dict(
    model=['pdnet'],
    **common_base_params,
)

pdnet_params = from_base_params_to_multiple_params(pdnet_params)

base_run_ids = [
    'ncpdnet_sense___rfs_radial_compound_mssim_dcomp_1611913984',
    'ncpdnet_sense___rfs_spiral_compound_mssim_dcomp_1611913984',
]
run_ids = (
    # classical eval
    duplicate_run_ids(base_run_ids) +
    # af 8 eval
    duplicate_run_ids(base_run_ids) +
    # reverse eval
    duplicate_run_ids(base_run_ids[::-1]) +
    # brain eval
    deduplicate_run_ids(base_run_ids)
)

pdnet_jobs = eval_grid(
    base_job_name + '_pdnet',
    eval_fun,
    pdnet_params,
    run_ids=run_ids,
    **grid_params,
)

#### Unet
unet_params = dict(
    model=['unet']
)

unet_params = from_base_params_to_multiple_params(unet_params)

base_run_ids = [
    'unet_mc___radial_compound_mssim_dcomp_1611915508',
    'unet_mc___spiral_compound_mssim_dcomp_1611915508',
]
run_ids = (
    # classical eval
    duplicate_run_ids(base_run_ids) +
    # af 8 eval
    duplicate_run_ids(base_run_ids) +
    # reverse eval
    duplicate_run_ids(base_run_ids[::-1]) +
    # brain eval
    deduplicate_run_ids(base_run_ids)
)


unet_jobs = eval_grid(
    base_job_name + '_unet',
    eval_fun,
    unet_params,
    run_ids=run_ids,
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
                multicoil=True,
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
jobs = pdnet_jobs + unet_jobs
params = extend_params(pdnet_params) + extend_params(unet_params)
for param, job in zip(params, jobs):
    metrics_names, eval_res = job.result()
    print('Parameters', param)
    print(eval_res)
