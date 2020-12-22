from fastmri_recon.data.scripts.postproc_volumes_tf_records_generation import generate_postproc_tf_records
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs

from jean_zay.submitit.general_submissions import get_executor


run_ids = {
    4: 'xpdnet_sense__af4_compound_mssim_rf_smb_MWCNNmedium_1606491318',
    8: 'xpdnet_sense__af8_compound_mssim_rf_smb_MWCNNmedium_1606491318',
}

model_name = 'MWCNN'
model_size = 'medium'
n_primal = 5
model_specs = list(get_model_specs(force_res=False, n_primal=n_primal))
if model_name is not None:
    model_specs = [ms for ms in model_specs if ms[0] == model_name]
if model_size is not None:
    model_specs = [ms for ms in model_specs if ms[1] == model_size]
_, model_size, model_fun, kwargs, _, n_scales, res = model_specs[0]
executor = get_executor('postproc_tfrecords', timeout_hour=20, n_gpus=1, project='fastmri')
with executor.batch():
    for mode in ['train', 'val']:
        for af in [4, 8]:
            executor.submit(
                generate_postproc_tf_records,
                af=af,
                mode=mode,
                model_fun=model_fun,
                model_kwargs=kwargs,
                run_id=run_ids[af],
                brain=False,
                n_epochs=300,
                n_iter=10,
                res=res,
                n_scales=n_scales,
                n_primal=n_primal,
                refine_smaps=True,
                refine_big=True,
                primal_only=False,
                n_dual=1,
                n_dual_filters=16,
            )
