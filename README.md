# A Brain-Inspired Regularizer for Adversarial Robustness

This repository is the official implementation of [**A Brain-Inspired Regularizer for Adversarial Robustness**][https://arxiv.org/abs/2410.03952]. 

It is based on the codebase developed by Li et al. in [Learning from brains how to regularize machines](https://arxiv.org/abs/1911.05072).
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
Training on grayscale images is the default setting in this codebase. To train and regularize on color images add the argument `rgb=True`.

### Training without regularization
To train a ResNet18 to classify CIFAR-10 without regularization, execute:

```
python train.py --task=CIFAR10 --archi=ResNet18
```

### Train with Regularization

To train a ResNet18 to classify CIFAR-10 regularized on images from CIFAR-10 using $S^{Th}$ as targets, execute:

```
python train.py --task=CIFAR10 --archi=ResNet18 --reg_data=CIFAR10 --reg_alpha=10 --reg_thresh=0.8
```
### Hyperparameters Used

| Classification - Regularization | $\alpha$ | $Th$ |
|---------------------------------|-----------|------|
| CIFAR-10 - CIFAR-10             | 10        | 0.8  |
| CIFAR-10 - CIFAR-100            | 10        | 0.8  |
| CIFAR-10 - ImageNet             | 10        | 0.8  |
| CIFAR-100 - CIFAR-10            | 10        | 0.8  |
| CIFAR-100 - CIFAR-100           | 10        | 0.8  |
| CIFAR-100 - ImageNet            | 10        | 0.8  |
| MNIST - MNIST                   | 4         | 0.2  |
| MNIST - FashionMNIST            | 10        | 0.8  |
| FashionMNIST - MNIST            | 10        | 0.4  |
| FashionMNIST - FashionMNIST     | 10        | 0.8  |

# Terminology

The argument names associated with the hyperparameters presented in the paper are as follows:

- **Number of regularization images $N$**: `--reg_image_num`
  
- **Regularization batch size $k$**: `--reg_batch_size`
  - Note: This parameter considers the number of pairs passed through the regularization pathway, so the actual regularization batch size is $2\times k$.

- **Range of similarities regularized on**: `--reg_range`
  - For $S_{low}^{Th}$: add `--reg_range=low`
  - For $S_{high}^{Th}$: add `--reg_range=high`
  - For $S_{+}^{Th}$: add `--reg_range=positive`
  - For $S_{-}^{Th}$: add `--reg_range=negative`
  - For $S_{double}^{Th_1, Th_2}$: add `--reg_range=double --reg_thresh_2=Th_2` where you can choose `Th_2` and `Th_1` corresponds to $Th$ in `--reg_thresh`.



## Adversarial Robustness Evaluation

Evaluation of models is automatically performed at the end of training, with the argument `--attack=True` set by default.

