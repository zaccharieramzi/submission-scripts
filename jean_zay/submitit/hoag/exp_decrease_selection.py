from hoag.scripts.best_exp_decrease import get_alpha_for_exp_decrease
import numpy as np

from jean_zay.submitit.general_submissions import get_executor


executor = get_executor('step_size_schedule', timeout_hour=2, n_gpus=1, project='hoag')

jobs = []
with executor.batch():
    for exp_decrease in np.linspace(0.6, 0.9, 20):
        job = executor.submit(
            get_alpha_for_exp_decrease,
            exp_decrease=exp_decrease,
        )
        jobs.append(job)

for job, exp_decrease in zip(jobs, np.linspace(0.6, 0.9, 20)):
    alpha, val_losses = job.result()
    print(exp_decrease, alpha, min(val_losses))