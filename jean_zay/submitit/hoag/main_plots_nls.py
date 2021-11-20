from hoag.scripts.main_plots_nls import run_scheme
import numpy as np

from jean_zay.submitit.general_submissions import get_cpu_executor


executor = get_cpu_executor('main_plots_nls_indep', timeout_hour=2, n_cpus=3, project='hoag')
scheme_labels = ['shine-big-rank', 'fpn', 'original']


jobs = []
with executor.batch():
    for scheme_label in scheme_labels:
        job = executor.submit(
            run_scheme,
            scheme_label=scheme_label,
        )
        jobs.append(job)

big_df_res = None
for job, exp_decrease in zip(jobs, scheme_labels):
    df_res = job.result()
    if big_df_res is None:
        big_df_res = df_res
    else:
        big_df_res = big_df_res.append(df_res)
big_df_res.to_csv('plots_nls_results.csv')
