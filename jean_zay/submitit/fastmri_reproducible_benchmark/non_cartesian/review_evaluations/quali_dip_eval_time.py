from fastmri_recon.evaluate.scripts.dip_qualitative_validation import dip_qualitative_validation

from jean_zay.submitit.general_submissions import get_executor

executor = get_executor('dip_sc_time', timeout_hour=2, n_gpus=1, project='fastmri4')
job = executor.submit(
    dip_qualitative_validation,
    model_kwargs={'bn': True, 'n_up': 5},
    acq_type='radial',
    af=4,
    multicoil=False,
    contrast='CORPD_FBK',
    n_iter=1000,
    timing=True,
)
duration = job.result()
print(duration)
