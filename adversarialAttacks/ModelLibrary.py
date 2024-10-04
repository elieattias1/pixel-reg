from enum import Enum


class ModelLibrary(Enum):
    # add the training id of unregularized models used for transferred FGSM attack on different datasets
    MNIST = ""  
    FASHION = ""
    CIFAR10 = ""
    CIFAR100 = ""
