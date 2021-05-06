from learning_wavelets.training_scripts.exact_recon_unet_training import train_unet
from learning_wavelets.evaluation_scripts.exact_recon_unet_evaluate import evaluate_unet

import numpy as np
from scipy.stats import ttest_ind

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'exact_recon_unet_training'
n_epochs = 100
batch_size = 8
base_n_filters = 4
n_layers = 4 
non_linearity = 'relu'
n_gpus = 1
possible_std_dev = [0.0001, 5, 15, 20, 25, 30, 50, 55, 60, 75]

parameters = dict(
    n_epochs=n_epochs,
    batch_size=batch_size,
    base_n_filters=base_n_filters,
    n_layers=n_layers,
    non_linearity=non_linearity 
)


res_all = train_eval_grid(
    job_name,
    train_unet,
    evaluate_unet,
    parameters,
    to_grid=False,
    timeout_train=2,
    n_gpus_train=n_gpus,
    timeout_eval=1,
    n_gpus_eval=n_gpus,
    project='exact_recon_unet',
    params_to_ignore=['n_epochs'],
    noise_std_test=possible_std_dev, 
)

print('Results')
print(res_all)
