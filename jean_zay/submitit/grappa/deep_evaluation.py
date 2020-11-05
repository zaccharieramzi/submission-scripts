from grappa.deep.model import DeepKSpaceFiller
from grappa.evaluate.deep_evaluation import test_model

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid

job_name = 'd_grappa_eval'
metrics = dict()
param_grid = {
    'model_fun': [DeepKSpaceFiller],
    'model_kwargs': [{'n_dense': 2}, {'n_dense': 3}],
    'distance_from_center_feat': [True, False],
    'n_epochs': [1000],
    'lr': [1e-3],
    'instance_normalisation': [True, False],
    'kernel_learning': [True, False],
}

eval_grid(
    job_name,
    test_model,
    param_grid,
    n_samples=2,
    timeout=2,
    n_gpus=1,
)
