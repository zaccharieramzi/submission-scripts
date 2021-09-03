from gan2gan.training.scripts.generator_training import wgan_training

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('wgan_training', timeout_hour=2, n_gpus=1, project='gan2gan')

job = executor.submit(
    wgan_training,
    run_id='test',
    epochs=1,
    N=4,
    patch_size=64,
    batch_size=4,
)