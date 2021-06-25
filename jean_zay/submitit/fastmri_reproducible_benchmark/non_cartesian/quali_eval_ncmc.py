from fastmri_recon.evaluate.scripts.qualitative_validation import nc_multinet_qualitative_validation

from jean_zay.submitit.general_submissions import get_executor


model_2_run_ids = {
    'pdnet': {
        'radial': 'ncpdnet_sense___rfs_radial_compound_mssim_dcomp_1611913984',
        'spiral': 'ncpdnet_sense___rfs_spiral_compound_mssim_dcomp_1611913984',
    },
    'unet': {
        'radial': 'unet_mc___radial_compound_mssim_dcomp_1611915508',
        'spiral': 'unet_mc___spiral_compound_mssim_dcomp_1611915508',
    },
    'adj-dcomp': {
        'radial': None,
        'spiral': None,
    },
}


executor = get_executor('ncmc_quali', timeout_hour=2, n_gpus=1, project='fastmri4')
with executor.batch():
    for model, run_ids in model_2_run_ids.items():
        for acq_type in ['radial', 'spiral']:
            executor.submit(
                nc_multinet_qualitative_validation,
                acq_type=acq_type,
                af=4,
                model=model,
                run_id=run_ids[acq_type],
                multicoil=True,
                refine_smaps=True,
                dcomp=True,
                normalize_image=False,
                contrast='CORPD_FBK',
                n_epochs=100,
                # zoom=[(200, 300), (100, 200)],
                draw_zoom=[(200, 300), (100, 200)],
            )
