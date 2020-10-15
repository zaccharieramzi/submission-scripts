from sklearn.model_selection import ParameterGrid
import submitit

from fastmri_recon.training_scripts.nc_train import train_ncpdnet, train_vnet_nc, train_unet_nc

def add(a, b):
    return a + b

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_test")
# set timeout in min, and partition for running the job
executor.update_parameters(timeout_min=1, slurm_partition="dev")
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()

def train_ncnet_submitit(
        af=4,
        n_epochs=70,
        loss='mae',
        refine_smaps=False,
        multicoil=False,
        model='pdnet',
        acq_type='radial',
        three_d=False,
        dcomp=False,
    ):
    if model == 'pdnet':
        train_function = train_ncpdnet
        add_kwargs = {'refine_smaps': refine_smaps}
    elif model == 'unet':
        if three_d:
            train_function = train_vnet_nc
        else:
            train_function = train_unet_nc
        add_kwargs = {}
    if multicoil:
        add_kwargs.update(dcomp=True)
    else:
        add_kwargs.update(dcomp=dcomp)
    train_function(
        af=af,
        n_epochs=n_epochs,
        loss=loss,
        multicoil=multicoil,
        acq_type=acq_type,
        three_d=three_d,
        **add_kwargs,
    )

def train_eval_grid(job_name, train_function, eval_function, parameter_grid):
    parameters = list(ParameterGrid(parameter_grid))
    executor = submitit.AutoExecutor(folder=job_name)
    executor.update_parameters(
        # TODO: see which args to change
        # cores=1,
        # job_cpu=20,
        # memory='80GB',
        # job_name=job_name,
        # walltime='60:00:00',
        # interface='ib0',
        # job_extra=[
        #     f'--gres=gpu:1',
        #     '--qos=qos_gpu-t4',
        #     '--distribution=block:block',
        #     '--hint=nomultithread',
        #     '--output=%x_%j.out',
        # ],
        # env_extra=[
        #     'cd $WORK/fastmri-reproducible-benchmark',
        #     '. ./submission_scripts_jean_zay/env_config.sh',
        # ],
        timeout_min=60 * 60,
        gpus_per_node=1,
        slurm_partition="dev",
    )
