# Instantiate the torchvision CIFAR10 dataset
# in order to download it
# this should be run on a front node with internet access

import torchvision.datasets as datasets

# Define the path to the dataset
train_dataset = datasets.CIFAR10(root='data/cifar10/', train=True, download=True, transform=None)
valid_dataset = datasets.CIFAR10(root='data/cifar10/', train=False, download=True, transform=None)
