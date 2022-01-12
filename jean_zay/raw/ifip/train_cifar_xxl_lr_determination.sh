#!/bin/bash

IFS=''
params=(
    '{lr_latent:1e-3,lr_mlp:1e-5},'
    '{lr_latent:5e-4,lr_mlp:5e-4},'
    '{lr_latent:5e-4,lr_mlp:1e-5},'
    '{lr_latent:1e-5,lr_mlp:1e-5},'
    '{lr_latent:5e-4,lr_mlp:5e-6},'
    '{lr_latent:1e-5,lr_mlp:5e-6}'
)

cd $WORK/implicit-fields-inverse-problems
submitit-hydra-launch ifip/training/train_cifar.py 4gpus_dev\
    +model=xxl training.fit.epochs=5\
    +model.mlp.skip_buffer=true\
    hydra.job.name='lr_buf_grid_search'\
    'callbacks.model_name=ifip_xxl_buf_lrl${compile.lr_latent}_lrm${compile.lr_mlp}'\
    "+compile=${params[*]}"