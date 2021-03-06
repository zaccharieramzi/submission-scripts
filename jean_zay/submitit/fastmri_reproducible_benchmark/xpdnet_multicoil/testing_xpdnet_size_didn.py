from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs

from jean_zay.submitit.general_submissions import get_executor
from jean_zay.submitit.fastmri_reproducible_benchmark.mem_fitting_test import test_works_in_xpdnet_train

n_iter_to_try_for_size = {
    'big': range(20, 30),
}
n_primal = 5


job_name = 'xpdnet_tryouts'
executor = get_executor(job_name, timeout_hour=1, n_gpus=1, project='fastmri4')

results = {}

with executor.batch():
    for data_consistency_learning in [False]:
        for model_size_spec, n_iter_to_try in n_iter_to_try_for_size.items():
            for model_name, model_size, model_fun, model_kwargs, n_inputs, n_scales, res in get_model_specs(n_primal):
                if model_size != model_size_spec or not (model_name in ['DIDN']):
                    continue
                for n_iter in n_iter_to_try:
                    job = executor.submit(
                        test_works_in_xpdnet_train,
                        model_fun=model_fun,
                        model_kwargs=model_kwargs,
                        n_scales=n_scales,
                        res=res,
                        n_iter=n_iter,
                        multicoil=True,
                        use_mixed_precision=True,
                        data_consistency_learning=data_consistency_learning,
                    )
                    res_id = (model_name, model_size, data_consistency_learning)
                    results[res_id] = results.get(res_id, [])
                    results[res_id].append(job)


for k, jobs in results.items():
    list_res = []
    for j in jobs:
        try:
            list_res.append(j.result())
        except:
            list_res.append(False)
    print(k, list_res)
