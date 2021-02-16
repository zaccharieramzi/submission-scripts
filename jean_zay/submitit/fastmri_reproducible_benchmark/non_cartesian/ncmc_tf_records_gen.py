from fastmri_recon.data.scripts.multicoil_nc_tf_records_generation import generate_multicoil_nc_tf_records

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('ncmc_tfrecords', timeout_hour=20, n_gpus=1, project='fastmri')
with executor.batch():
    for mode in ['train', 'val']:
        for acq_type in ['radial', 'spiral']:
            executor.submit(generate_multicoil_nc_tf_records, acq_type=acq_type, af=4, mode=mode, brain=True)
