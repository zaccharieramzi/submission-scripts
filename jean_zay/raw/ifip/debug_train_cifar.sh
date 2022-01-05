cd $WORK/implicit-fields-inverse-problems
python ifip/training/train_cifar.py -i -m hydra/launcher=dev\
    'hydra.searchpath=[pkg://jean_zay/hydra_config]'\
    training.fit.epochs=1 note=debugging\
    '+compile={lr_latent:5e-4,lr_mlp:1e-5}'