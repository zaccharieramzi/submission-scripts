from timm.data import create_dataset, create_loader

dataset_train = create_dataset(
    'tfds/imagenet2012',
    root="/gpfsscratch/rech/xpa/uap69lx/tensorflow_datasets", split='train', is_training=True,
    batch_size=128, repeats=1)

timm_loader = create_loader(
    dataset_train,
    input_size=(3, 224, 224),
    batch_size=128,
    is_training=True,
    use_prefetcher=True,
    no_aug=False,
    scale=[0.08, 1.0],
    ratio=[3./4., 4./3.],
    hflip=0.5,
    vflip=0.,
    color_jitter=0.4,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    num_workers=10,
    pin_memory=True,
    persistent_workers=True,
)

import time

counter = 0
for X, y in timm_loader:
    X, y = X.cuda(), y.cuda()
    time.sleep(0.1)
    counter += 1
    print(counter)
    if counter > 250:
        break
