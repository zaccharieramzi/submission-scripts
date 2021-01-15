import tensorflow as tf

from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.models.training.compile import default_model_compile


n_primal = 5

def test_works_in_xpdnet_train(model_fun, model_kwargs, n_scales, res, n_iter=10):
    run_params = {
        'n_primal': n_primal,
        'multicoil': False,
        'n_scales': n_scales,
        'n_iter': n_iter,
        'refine_smaps': False,
        'res': res,
    }
    model = XPDNet(model_fun, model_kwargs, **run_params)
    default_model_compile(model, lr=1e-3, loss='mae')
    base_shape = 320
    try:
        model.fit(
            x=[
                tf.ones([1, base_shape, base_shape, 1], dtype=tf.complex64),
                tf.ones([1, base_shape, base_shape], dtype=tf.complex64),
            ],
            y=tf.ones([1, 320, 320, 1]),
            epochs=1,
        )
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError):
        return False
    else:
        return True
