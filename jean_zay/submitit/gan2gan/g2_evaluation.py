from gan2gan.config import CKPT_PATH
from gan2gan.evaluate.eval import eval_wgan

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('wgan_eval', timeout_hour=2, n_gpus=1, project='gan2gan')

job = executor.submit(
    eval_wgan,
    ckpt_path=CKPT_PATH / 'test-30',
    use_g2=True,
    name='test',
)