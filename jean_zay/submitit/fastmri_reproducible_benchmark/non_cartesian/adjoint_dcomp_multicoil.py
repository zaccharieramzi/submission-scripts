from fastmri_recon.evaluate.scripts.nc_eval import evaluate_dcomp

from jean_zay.submitit.general_submissions import get_executor

executor = get_executor('adjoint_dc_mc', timeout_hour=20, n_gpus=1, project='fastmri4')
with executor.batch():
    for acq_type in ['radial', 'spiral']:
        executor.submit(
            evaluate_dcomp,
            acq_type=acq_type,
            af=4,
            n_samples=100,
            multicoil=True,
        )
