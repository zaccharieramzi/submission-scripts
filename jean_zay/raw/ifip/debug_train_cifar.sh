#!/bin/bash

params=(
    '{lr_latent:5e-4,lr_mlp:1e-5},'
    '{lr_latent:1e-5,lr_mlp:5e-6}'
)

cd $WORK/implicit-fields-inverse-problems
python ifip/training/train_cifar.py -m hydra/launcher=dev\
    'hydra.searchpath=[pkg://jean_zay/hydra_config]'\
    training.fit.epochs=1 note=debugging\
    +compile=${params[@]}