from fastmri_recon.models.subclassed_models.feature_level_multi_domain_learning.proposed_params import get_model_specs

from jean_zay.submitit.general_submissions import get_executor
from jean_zay.submitit.fastmri_reproducible_benchmark.mem_fitting_test import test_works_in_xpdnet_train

n_iter_to_try_for_size = {
    # 'small': list(range(20, 25)),
    # 'medium': list(range(15, 20)),
    # 'big': list(range(5, 15)),
    'XL': list(range(1, 5)),
}
n_primal = 5


job_name = 'flmd_tryouts'
executor = get_executor(job_name, timeout_hour=6, n_gpus=1, project='mdeq')

results = {}

with executor.batch():
    for model_size_spec, n_iter_to_try in n_iter_to_try_for_size.items():
        for n_iter in n_iter_to_try:
            for model_name, model_size, model_fun, model_kwargs, n_inputs, n_scales, res in get_model_specs(n_primal):
                if model_size != model_size_spec:
                    continue
                job = executor.submit(
                    test_works_in_xpdnet_train,
                    model_fun=model_fun,
                    model_kwargs=model_kwargs,
                    n_scales=n_scales,
                    res=res,
                    n_iter=n_iter,
                )
                results[(model_name, model_size)] = results.get((model_name, model_size), [])
                results[(model_name, model_size)].append(job)


for k, jobs in results.items():
    print(k, [j.result() for j in jobs])
