from fastmri_recon.data.scripts.qualitative_validation import nc_multinet_qualitative_validation

from jean_zay.submitit.general_submissions import get_executor


model_2_run_ids = {
    'pdnet': {
        'radial': 'ncpdnet_3d___i6_radial_stacks_mse_dcomp_1612291359',
        'spiral': 'ncpdnet_3d___i6_spiral_stacks_mse_dcomp_161229135911913984',
    },
    'unet': {
        'radial': 'vnet_3d___radial_stacks_mse_dcomp_1612291357',
        'spiral': 'vnet_3d___spiral_stacks_mse_dcomp_1612291357',
    },
    'adj-dcomp': {
        'radial': None,
        'spiral': None,
    },
}


executor = get_executor('3dnc_quali', timeout_hour=20, n_gpus=1, project='fastmri4')
with executor.batch():
    for model, run_ids in model_2_run_ids.items():
        for acq_type in ['radial_stacks', 'spiral_stacks']:
            executor.submit(
                nc_multinet_qualitative_validation,
                acq_type=acq_type,
                af=4,
                model=model,
                run_id=run_ids[acq_type],
                three_d=True,
                refine_smaps=False,
                dcomp=True,
                normalize_image=False,
                n_epochs=8,
            )
