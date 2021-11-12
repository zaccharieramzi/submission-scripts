from fastmri_recon.data.scripts.multicoil_nc_tf_records_generation import generate_multicoil_nc_tf_records

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('ncmc_tfrecords', timeout_hour=20, n_gpus=4, project='fastmri4', no_force_32=True)
with executor.batch():
    # for mode in ['train', 'val']:
    for mode in ['val']:
        for acq_type in ['radial', 'spiral']:
            for af in [4, 8]:
                for brain in [True, False]:
                    executor.submit(generate_multicoil_nc_tf_records, acq_type=acq_type, af=af, mode=mode, brain=brain)
