cd $WORK/implicit-fields-inverse-problems
python ifip/training/train_cifar.py --multirun hydra/launcher=dev 'hydra.searchpath=[pkg://jean_zay/hydra_config]' training.fit.epochs=1 note=debugging