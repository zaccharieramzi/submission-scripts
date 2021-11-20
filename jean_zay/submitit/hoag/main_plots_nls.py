from hoag.scripts.main_plots_nls import run_scheme
import numpy as np

from jean_zay.submitit.general_submissions import get_cpu_executor


executor = get_cpu_executor('main_plots_nls_indep', timeout_hour=2, n_cpus=10, project='hoag')
scheme_labels = ['shine-big-rank', 'fpn', 'original', 'shine-big-rank-refined', 'shine-big-rank-opa']
# search_space = np.linspace(0.75, 0.85, 10)
search_space = [0.8]

jobs = []
with executor.batch():
    for exponential_decrease_factor in search_space:
        for scheme_label in scheme_labels:
            job = executor.submit(
                run_scheme,
                scheme_label=scheme_label,
                exponential_decrease_factor=exponential_decrease_factor,
            )
            jobs.append(job)

job_counter = 0
for exponential_decrease_factor in search_space:
    big_df_res = None
    for scheme_label in scheme_labels:
        job = jobs[job_counter]
        job_counter += 1
        df_res = job.result()
        if big_df_res is None:
            big_df_res = df_res
        else:
            big_df_res = big_df_res.append(df_res)
    big_df_res.to_csv(f'plots_nls_results_exp{exponential_decrease_factor}.csv')
