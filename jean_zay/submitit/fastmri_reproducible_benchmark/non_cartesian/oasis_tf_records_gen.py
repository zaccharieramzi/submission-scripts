from fastmri_recon.data.scripts.oasis_tf_records_generation import generate_oasis_tf_records

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('oasis_tfrecords', timeout_hour=20, n_gpus=1, project='fastmri4', no_force_32=True)
with executor.batch():
    for mode in ['train', 'val']:
        # for acq_type in ['radial_stacks', 'spiral_stacks']:
        for acq_type in ['radial']:
            executor.submit(generate_oasis_tf_records, acq_type=acq_type, af=4, mode=mode)
