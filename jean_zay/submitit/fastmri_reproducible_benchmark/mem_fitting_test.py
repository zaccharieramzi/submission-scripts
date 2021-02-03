import tensorflow as tf
from tensorflow.keras import mixed_precision

from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.models.training.compile import default_model_compile


n_primal = 5

def test_works_in_xpdnet_train(model_fun, model_kwargs, n_scales, res, n_iter=10, multicoil=False, use_mixed_precision=False, data_consistency_learning=False):
    # trying mixed precision
    if use_mixed_precision:
        policy_type = 'mixed_float16'
    else:
        policy_type = 'float32'
    mixed_precision.set_global_policy(policy_type)
    run_params = {
        'n_primal': n_primal,
        'multicoil': multicoil,
        'n_scales': n_scales,
        'n_iter': n_iter,
        'refine_smaps': multicoil,
        'res': res,
        'primal_only': not data_consistency_learning,
    }
    model = XPDNet(model_fun, model_kwargs, **run_params)
    default_model_compile(model, lr=1e-3, loss='mae')
    base_shape = 640
    n_coils = 15
    k_shape = (base_shape,)*2
    if multicoil:
        k_shape = (n_coils, *k_shape)
    inputs = [
        tf.ones([1, *k_shape, 1], dtype=tf.complex64),
        tf.ones([1, *k_shape], dtype=tf.complex64),
    ]
    if multicoil:
        inputs += [tf.ones([1, *k_shape], dtype=tf.complex64),]
    try:
        model.fit(
            x=inputs,
            y=tf.ones([1, 320, 320, 1]),
            epochs=1,
        )
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError):
        return False
    else:
        return True
