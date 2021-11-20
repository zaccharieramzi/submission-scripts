from hoag.scripts.best_exp_decrease import get_alpha_for_exp_decrease
import numpy as np

from jean_zay.submitit.general_submissions import get_cpu_executor


executor = get_cpu_executor('step_size_schedule', timeout_hour=2, n_gpus=1, project='hoag')
search_space = np.linspace(0.75, 0.85, 80)


jobs = []
with executor.batch():
    for exp_decrease in search_space:
        job = executor.submit(
            get_alpha_for_exp_decrease,
            exp_decrease=exp_decrease,
            max_iter=100,
        )
        jobs.append(job)

for job, exp_decrease in zip(jobs, search_space):
    alpha, val_losses = job.result()
    print(exp_decrease, alpha, min(val_losses))