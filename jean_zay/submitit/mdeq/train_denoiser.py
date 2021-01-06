from mdeq_lib.training_scripts.denoise_train import train_mdeq_denoising

from jean_zay.submitit.general_submissions import get_executor


job_name = 'mdeq_denoise'
executor = get_executor(job_name, timeout_hour=20, n_gpus=1, project='mdeq')

executor.submit(
    train_mdeq_denoising,
    n_val=20,
)
