from mdeq_lib.training_scripts.denoise_train import train_mdeq_denoising

from jean_zay.submitit.general_submissions import get_executor


job_name = 'mdeq_denoise'
executor = get_executor(job_name, timeout_hour=6, n_gpus=1, project='mdeq')

with executor.batch():
    for use_bn in [True, False]:
        for use_res in [True, False]:
            executor.submit(
                train_mdeq_denoising,
                n_val=20,
                use_res=use_res,
                use_bn=use_bn,
                network_size='SMALL',
            )
