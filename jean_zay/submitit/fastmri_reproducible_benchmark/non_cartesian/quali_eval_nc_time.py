from fastmri_recon.evaluate.scripts.qualitative_validation import nc_multinet_qualitative_validation

from jean_zay.submitit.general_submissions import get_executor


model_2_run_ids = {
    'pdnet': {
        'radial': 'ncpdnet_singlecoil___radial_compound_mssim_dcomp_1610872636',
        'spiral': 'ncpdnet_singlecoil___spiral_compound_mssim_dcomp_1610911070',
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

pdnet_normed = {
    'radial': 'ncpdnet_singlecoil___radial_compound_mssim_1610720754',
    'spiral': 'ncpdnet_singlecoil___spiral_compound_mssim_1610720754',
}

executor = get_executor('nc_time', timeout_hour=2, n_gpus=1, project='fastmri4')
with executor.batch():
    for model, run_ids in model_2_run_ids.items():
        for acq_type in ['radial', 'spiral']:
            executor.submit(
                nc_multinet_qualitative_validation,
                acq_type=acq_type,
                af=4,
                model=model,
                run_id=run_ids[acq_type],
                multicoil=False,
                refine_smaps=False,
                dcomp=True,
                normalize_image=False,
                contrast='CORPD_FBK',
                n_epochs=100,
                timing=True,
            )
    for acq_type in ['radial', 'spiral']:
        executor.submit(
            nc_multinet_qualitative_validation,
            acq_type=acq_type,
            af=4,
            model='pdnet',
            run_id=pdnet_normed[acq_type],
            multicoil=False,
            refine_smaps=False,
            dcomp=False,
            normalize_image=True,
            contrast='CORPD_FBK',
            n_epochs=100,
            timing=True,
        )
