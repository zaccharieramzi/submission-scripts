from mdeq_lib.training.cls_train import train_classifier


from jean_zay.submitit.general_submissions import train_eval_grid


job_name = 'shine_classifier_imagenet_small_pretrain'
n_gpus = 4
n_runs = 1
base_params = dict(
    model_size='SMALL',
    dataset='imagenet',
    n_gpus=n_gpus,
    n_epochs=100,
    save_at=[50, 60, 70, 80],
    restart_from=50,
)
parameters = []
for i_run in range(n_runs):
    parameters += [
        dict(seed=i_run, **base_params),
        # dict(seed=i_run, shine=True, **base_params),
        # dict(seed=i_run, fpn=True, **base_params),
    ]

def _dummy_eval(**kwargs):
    pass

res_all = train_eval_grid(
    job_name,
    train_classifier,
    _dummy_eval,
    parameters,
    to_grid=False,
    timeout_train=100,
    n_gpus_train=n_gpus,
    timeout_eval=0,
    n_gpus_eval=0,
    project='shine',
    params_to_ignore=['n_epochs', 'save_at'],
    torch=True,
    force_partition='gpu_p2',
)

print(res_all)
