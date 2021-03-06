from fastmri_recon.evaluate.scripts.dip_qualitative_validation import dip_qualitative_validation

from jean_zay.submitit.general_submissions import get_executor


zoom_box = [(200, 300), (100, 200)]

executor = get_executor('dip_mc_brain_quali', timeout_hour=2, n_gpus=1, project='fastmri4')
with executor.batch():
    for acq_type in ['radial', 'spiral']:
        for af in [4]:
            executor.submit(
                dip_qualitative_validation,
                model_kwargs={'bn': True, 'n_up': 5},
                acq_type=acq_type,
                af=af,
                contrast=None,
                slice_index=5,
                n_iter=1000,
                # zoom=zoom_box,
                draw_zoom=zoom_box,
                multicoil=True,
                brain=True,
            )
