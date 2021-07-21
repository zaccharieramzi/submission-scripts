from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.training_scripts.xpdnet_train_block import train_xpdnet_block
from fastmri_recon.evaluate.scripts.xpdnet_eval import evaluate_xpdnet

from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'xpdnet_stacked_sc'
loss = 'compound_mssim'
lr = 1e-4
batch_size = None
n_samples = None
n_epochs = 300
n_primal = 5
contrast = None
primal_only = True

model_name = 'DnCNN'
model_size = 'small'
model_specs = list(get_model_specs(n_primal=n_primal))
model_specs = [(m_name, m_size, *x) for (m_name, m_size, *x) in model_specs if m_name == model_name and m_size == model_size]
model_name, model_size, model_fun, kwargs, _, n_scales, res = model_specs[0]

base_parameters = dict(
    loss=[loss],
    lr=[lr],
    batch_size=[batch_size],
    n_samples=[n_samples],
    epochs_per_block_step=[n_epochs],
    n_primal=[n_primal],
    contrast=[contrast],
    primal_only=[primal_only],
    model_size=[model_size],
    model_fun=[model_fun],
    model_kwargs=[kwargs],
    multicoil=[False],
)

parameters = [
    # classical training
    dict(n_iter=[6], n_epochs=[n_epochs], **base_parameters),
    # stacked training
    dict(n_iter=[12], n_epochs=[2*n_epochs], **base_parameters),
    # stacked training with overlap
    dict(n_iter=[12], block_overlap=[6], n_epochs=[3*n_epochs], **base_parameters),
]

eval_results = train_eval_grid(
    job_name,
    train_xpdnet_block,
    evaluate_xpdnet,
    parameters,
    n_samples_eval=50,
    timeout_train=100,
    n_gpus_train=1,
    timeout_eval=10,
    n_gpus_eval=1,
    params_to_ignore=['batch_size', 'block_overlap', 'model_size', 'epochs_per_block_step', 'block_size'],
)

print(eval_results)