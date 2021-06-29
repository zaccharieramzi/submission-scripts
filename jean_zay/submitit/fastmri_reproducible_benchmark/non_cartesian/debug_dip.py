import pandas as pd
from fastmri_recon.evaluate.scripts.debug_nc_dip_eval import debug_dip_nc

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('debug_dip', timeout_hour=2, n_gpus=1, project='fastmri4')
job = executor.submit(
    debug_dip_nc,
    acq_type='radial',
    af=4,
    contrast='CORPD_FBK',
    model_kwargs={'bn': True},
    n_iter=10,
)

save_path = job.result()
print(save_path)
