from fastmri_recon.evaluate.scripts.nc_eval import evaluate_dcomp
from fastmri_recon.evaluate.scripts.nc_eval import evaluate_nc_multinet as eval_fun

from jean_zay.submitit.general_submissions import get_executor, eval_grid

grid_params = dict(
    n_samples=100,
    n_gpus=3,
    wait=False,
    no_force_32=False,
    project='fastmri4',
)

base_job_name = 'ncmc_eval'
contrasts_knee = ['CORPD_FBK', 'CORPDFS_FBK']
contrasts_brain = ['AXT1', 'AXT2', 'AXT1POST', 'AXT1PRE', 'AXFLAIR']
common_base_params = dict(
    af=[4],
    acq_type=['radial', 'spiral'],
    n_epochs=[100],
    multicoil=[True],
    refine_smaps=[True],
    dcomp=[True],
)

def from_base_params_to_multiple_params(base_params):
    multiple_params = [
        # classical eval
        dict(contrast=contrasts_knee, **base_params),
        # reversed eval
        dict(contrast=contrasts_knee, **base_params),
        # brain eval
        dict(contrast=contrasts_brain, brain=[True], **base_params)
    ]
    return multiple_params

def duplicate_run_ids(base_run_ids):
    new_run_ids = [run_id for pair in zip(base_run_ids, base_run_ids) for run_id in pair]
    return new_run_ids

def deduplicate_run_ids(base_run_ids):
    new_run_ids = [run_id for quintuple in zip(*([base_run_ids]*5)) for run_id in quintuple]
    return new_run_ids

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
run_ids = duplicate_run_ids(base_run_ids) + duplicate_run_ids(base_run_ids[::-1]) + deduplicate_run_ids(base_run_ids)

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
run_ids = duplicate_run_ids(base_run_ids) + duplicate_run_ids(base_run_ids[::-1]) + deduplicate_run_ids(base_run_ids)


unet_jobs = eval_grid(
    base_job_name + '_unet',
    eval_fun,
    unet_params,
    run_ids=run_ids,
    **grid_params,
)

#### Adj + DC
executor = get_executor(base_job_name + 'adjoint_dc', timeout_hour=10, n_gpus=1, project='fastmri4')
with executor.batch():
    for acq_type in common_base_params['acq_type']:
        for contrast in common_base_params['contrast']:
            job = executor.submit(
                evaluate_dcomp,
                acq_type=acq_type,
                af=common_base_params['af'][0],
                n_samples=common_base_params['n_samples'][0],
                contrast=contrast,
                multicoil=True,
            )
            metrics_names, eval_res = job.result()
            print('Parameters for Adj+DC', acq_type, contrast)
            print(eval_res)

### Results presentation
### Nets
jobs = pdnet_jobs + unet_jobs
params = pdnet_params + unet_params
for param, job in zip(params, jobs):
    metrics_names, eval_res = job.result()
    print('Parameters', param)
    print(eval_res)
