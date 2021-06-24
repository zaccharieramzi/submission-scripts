from fastmri_recon.evaluate.scripts.qualitative_validation import nc_multinet_qualitative_validation

from jean_zay.submitit.general_submissions import get_executor


model_2_run_ids = {
    'ncpdnet-dcomp': {
        'radial': 'ncpdnet_singlecoil___radial_compound_mssim_dcomp_1610872636',
        'spiral': 'ncpdnet_singlecoil___spiral_compound_mssim_dcomp_1610911070',
    },
    'ncpdnet': {
        'radial': 'ncpdnet_singlecoil___radial_compound_mssim_1610720754',
        'spiral': 'ncpdnet_singlecoil___spiral_compound_mssim_1610720754',
    },
    'pdnet-gridded': {
        'radial': 'pdnet_gridded_singlecoil___radial_compound_mssim_1611913690',
        'spiral': 'pdnet_gridded_singlecoil___spiral_compound_mssim_1611913692',
    },
    'unet': {
        'radial': 'unet_singlecoil___radial_compound_mssim_dcomp_1610911070',
        'spiral': 'unet_singlecoil___spiral_compound_mssim_dcomp_1610911070',
    },
    'adj-dcomp': {
        'radial': None,
        'spiral': None,
    },
}


executor = get_executor('ncsc_quali', timeout_hour=2, n_gpus=1, project='fastmri4')
with executor.batch():
    for model, run_ids in model_2_run_ids.items():
        if 'ncpdnet' in model:
            dcomp = 'dcomp' in model
            normalize_image = not dcomp
            model = 'pdnet'
        else:
            dcomp = True
            normalize_image = False
        for acq_type in ['radial', 'spiral']:
            executor.submit(
                nc_multinet_qualitative_validation,
                acq_type=acq_type,
                af=4,
                model=model,
                run_id=run_ids[acq_type],
                multicoil=False,
                refine_smaps=False,
                dcomp=dcomp,
                normalize_image=normalize_image,
                contrast='CORPD_FBK',
                n_epochs=100,
            )
