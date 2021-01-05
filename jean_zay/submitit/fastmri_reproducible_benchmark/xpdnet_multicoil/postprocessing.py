from fastmri_recon.evaluate.scripts.postprocess_eval import evaluate_vnet_postproc
# from fastmri_recon.evaluate.scripts.xpdnet_inference import xpdnet_inference
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.training_scripts.postprocess_train import train_vnet_postproc

from jean_zay.submitit.general_submissions import train_eval_grid, eval_grid, infer_grid


job_name = 'post_process'
run_ids = {
    4: 'xpdnet_sense__af4_compound_mssim_rf_smb_MWCNNmedium_1606491318',
    8: 'xpdnet_sense__af8_compound_mssim_rf_smb_MWCNNmedium_1606491318',
}
loss = 'compound_mssim'
brain = False
lr = 1e-4
n_samples = None
n_epochs = 190
use_mixed_precision = False

parameter_grid = [
    dict(
        original_run_id=run_ids[8],
        brain=brain,
        n_epochs=n_epochs,
        n_samples=n_samples,
        af=8,
        loss=loss,
        lr=lr,
        use_mixed_precision=use_mixed_precision,
    )
] + [
    dict(
        original_run_id=run_ids[4],
        brain=brain,
        n_epochs=n_epochs,
        n_samples=n_samples,
        af=4,
        loss=loss,
        lr=lr,
        use_mixed_precision=use_mixed_precision,
    )
]

eval_results, run_ids = train_eval_grid(
# run_ids = [
#     'xpdnet_sense__af8_compound_mssim_rf_smb_MWCNNmedium_1606491318',
#     'xpdnet_sense__af4_compound_mssim_rf_smb_MWCNNmedium_1606491318',
# ]

# eval_results = eval_grid(
    job_name,
    train_vnet_postproc,
    evaluate_vnet_postproc,
    parameter_grid,
    # run_ids=run_ids,
    n_samples_eval=50,
    timeout_train=100,
    n_gpus_train=1,
    timeout_eval=10,
    n_gpus_eval=1,
    # n_samples=100,
    # timeout=10,
    # n_gpus=1,
    to_grid=False,
    return_run_ids=True,
    params_to_ignore=['use_mixed_precision']
)

print(eval_results)

# infer_grid(
#     job_name,
#     xpdnet_inference,
#     parameter_grid,
#     run_ids=run_ids,
#     timeout=10,
#     n_gpus=4,
#     to_grid=False,
#     params_to_ignore=['mask_type', 'multicoil'],
# )
