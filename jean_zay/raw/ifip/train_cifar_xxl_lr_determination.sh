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
python ifip/training/train_cifar.py -m hydra/launcher=4gpus_dev\
    'hydra.searchpath=[pkg://jean_zay/hydra_config]'\
    +model=xxl training.fit.epochs=5\
    "+compile=${params[*]}"