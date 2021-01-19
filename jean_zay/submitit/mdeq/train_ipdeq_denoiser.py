from mdeq_lib.training_scripts.denoise_train import train_ipdeq_denoising

from jean_zay.submitit.general_submissions import get_executor


job_name = 'ipdeq_denoise'
executor = get_executor(job_name, timeout_hour=20, n_gpus=1, project='mdeq')

with executor.batch():
    for with_dc in [True, False]:
        executor.submit(
            train_ipdeq_denoising,
            n_val=20,
            network_size='SMALL',
            grad_clipping=100.,
            with_dc=with_dc,
        )
