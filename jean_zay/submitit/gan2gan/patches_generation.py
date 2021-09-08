from gan2gan.config import CKPT_PATH
from gan2gan.evaluate.noisy_patches_gen import gen_noisy_patches

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('noisy_patches', timeout_hour=2, n_gpus=1, project='gan2gan')

job = executor.submit(
    gen_noisy_patches,
    run_id=CKPT_PATH / 'test-30',
    name='test',
    n_patches=100,
)