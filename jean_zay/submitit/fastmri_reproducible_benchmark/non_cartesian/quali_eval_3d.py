from fastmri_recon.evaluate.scripts.qualitative_validation import nc_multinet_qualitative_validation

from jean_zay.submitit.general_submissions import get_executor


model_2_run_ids = {
    # 'pdnet': {
    # },
    # 'unet': {
    # },
    'adj-dcomp': {
        'radial': None,
    },
}

zoom_box = [(150, 200), (100, 200)]

executor = get_executor('3dnc_quali', timeout_hour=2, n_gpus=1, project='fastmri4')
with executor.batch():
    for model, run_ids in model_2_run_ids.items():
        for acq_type in ['radial']:
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
                n_filters=16,
                n_iter=6,
                n_primal=2,
                slice_index=100,
                # zoom=zoom_box,
                draw_zoom=zoom_box,
            )
