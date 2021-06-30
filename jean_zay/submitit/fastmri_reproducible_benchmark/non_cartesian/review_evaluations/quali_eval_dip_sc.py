from fastmri_recon.evaluate.scripts.dip_qualitative_validation import dip_qualitative_validation

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('dip_sc_quali', timeout_hour=2, n_gpus=1, project='fastmri4')
with executor.batch():
    for acq_type in ['radial', 'spiral']:
        for af in [4, 8]:
            executor.submit(
                dip_qualitative_validation,
                model_kwargs={'bn': True, 'n_up': 5},
                acq_type=acq_type,
                af=af,
                contrast='CORPD_FBK',
                n_iter=1000,
                # zoom=[(200, 300), (100, 200)],
                draw_zoom=[(200, 300), (100, 200)],
            )
