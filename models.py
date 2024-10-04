# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# densely connected convolutional core. each layer gets inputs from all previous
# layers, and all layers are concatenated as the final features. a number of scales
# are provided as outputs, on which spatial transformer will read a local patch
#
# core_channels: list of ints, feature number of each layer
# kernel_sizes: list of ints, kernerl sizes of each layer
# scale_num: int, number of scales
# scale_factor: float, scaling factor
# readout_resol: int, resolution of local patch to read from
# basis_num: int or None, number of shared linear readout basis, None indicating
# independent linear readout for all scans
# in_channels: int
# nonlinearity: nn.Module or None
class DenseCore(nn.Module):
    def __init__(
        self,
        core_channels,
        kernel_sizes,
        scale_num=6,
        scale_factor=0.5,
        readout_resol=1,
        basis_num=None,
        in_channels=1,
        nonlinearity=None,
    ):
        super(DenseCore, self).__init__()
        assert len(core_channels) == len(
            kernel_sizes
        ), "require same number of channels and kernel sizes"
        for k_size in kernel_sizes:
            assert k_size % 2 == 1, "all kernel sizes should be odd"

        self.scale_num, self.scale_factor = scale_num, scale_factor
        self.readout_resol = readout_resol

        self.in_channels = in_channels
        if nonlinearity is None:
            nonlinearity = nn.ELU()
        self.nonlinearity = nonlinearity

        self.layers = nn.ModuleList()
        for i, (out_c, k_size) in enumerate(zip(core_channels, kernel_sizes)):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if i == 0 else sum(core_channels[:i]),
                        out_c,
                        k_size,
                        padding=k_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_c),
                    nonlinearity,
                )
            )

        self.out_channels = sum(core_channels)
        self.patch_vol = self.out_channels * self.scale_num * (readout_resol**2)
        if basis_num is None:
            self.readout_basis = None
        else:
            self.readout_basis = nn.Linear(self.patch_vol, basis_num, bias=False)

    def laplace_reg(self, layer_num=1):
        lap_kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

        param_count, reg_loss = 0, 0.0
        for layer_idx in range(layer_num):
            conv_w = self.layers[layer_idx][0].weight

            param_count += np.prod(conv_w.shape)
            _, _, H, W = conv_w.shape
            reg_loss = (
                reg_loss
                + F.conv2d(conv_w.view(-1, 1, H, W), lap_kernel[None, None].to(conv_w))
                .pow(2)
                .sum()
            )
        return reg_loss / param_count

    def weight_reg(self):
        param_count, reg_loss = 0, 0.0
        for layer in self.layers:
            param_count += np.prod(layer[0].weight.shape)
            reg_loss = reg_loss + layer[0].weight.pow(2).sum()
        return reg_loss / param_count

    def forward(self, images):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                outputs.append(layer(images))
            else:
                outputs.append(layer(torch.cat(outputs, dim=1)))
        outputs = [torch.cat(outputs, dim=1)]
        for i in range(1, self.scale_num):
            outputs.append(
                F.interpolate(outputs[-1], scale_factor=self.scale_factor, mode="area")
            )
        return outputs


# ResNet block
# expansion = 1 represents simple block, otherwise bottleneck block
class ResBlock(nn.Module):
    def __init__(self, in_channels, base_channels, stride=1, expansion=1):
        super(ResBlock, self).__init__()
        out_channels = expansion * base_channels

        self.layer0 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=3 if expansion == 1 else 1,
                padding=1 if expansion == 1 else 0,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
        )

        self.layer1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                base_channels, base_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
        )

        self.layer2 = (
            nn.Sequential()
            if expansion == 1
            else nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(out + self.shortcut(x))
        return out


# ResNet backbone
#
# in_channels: int
# base_channels: int
# block_nums: list of ints, numbers of ResNet blocks of each section
# block_type: str, ResNet block type
# pool_size: int, pool size of the final pooling layer
class ResNetCore(nn.Module):
    def __init__(self, in_channels, base_channels, block_nums, block_type, pool_size):
        super(ResNetCore, self).__init__()

        assert block_type == "basic" or block_type == "bottleneck"
        self.section_num = len(block_nums)

        self.sections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, base_channels, kernel_size=7, padding=3, bias=False
                    ),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(),
                )
            ]
        )

        in_channels = base_channels
        base_channels = [base_channels * (2**i) for i in range(self.section_num)]
        strides = [1] + [2] * (self.section_num - 1)
        expansion = 1 if block_type == "basic" else 4

        for i in range(self.section_num):
            section, in_channels = self._make_section(
                block_nums[i], in_channels, base_channels[i], strides[i], expansion
            )
            self.sections.append(section)

        self.pool_size = pool_size

    def _make_section(self, block_num, in_channels, base_channels, stride, expansion):
        strides = [stride] + [1] * (block_num - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResBlock(in_channels, base_channels, stride, expansion))
            in_channels = expansion * base_channels
        return nn.Sequential(*blocks), in_channels

    # return features after each ResNet block, along with the one after first
    # convolutional layer and the one after final pooling
    def forward(self, x):
        features = [self.sections[0](x)]
        for i in range(self.section_num):
            features.append(self.sections[i + 1](features[i]))
        features.append(F.avg_pool2d(features[self.section_num], self.pool_size))
        return features

    # in_shape (tuple): (c, h, w)
    def feature_shapes(self, in_shape):
        inputs = torch.randn((1, *in_shape), device=self.sections[0][0].weight.device)
        features = self.forward(inputs)
        return [f.shape[1:] for f in features]


# ResNet that takes one batch or a pair of batches of images as inputs
#
# in_shape: tuple of ints, (c, h, w) for input images
# num_classes: int, number of classes
# base_channels: int
# block_nums: list of ints
# block_type: str
# pool_size: int
# beta: float, decay rate for updating running mean, used when calculating
# average feature activations
class ResNet(nn.Module):
    def __init__(
        self,
        in_shape,
        num_classes,
        base_channels,
        block_nums,
        block_type,
        pool_size=4,
        beta=0.9,
    ):
        super(ResNet, self).__init__()

        self.core = ResNetCore(
            in_shape[0], base_channels, block_nums, block_type, pool_size
        )

        f_shapes = self.core.feature_shapes(in_shape)  # shapes of all features returned
        self.linear = nn.Linear(np.prod(f_shapes[-1]), num_classes)

        self.layer_weight = nn.Parameter(
            torch.zeros(self.core.section_num + 1, dtype=torch.float)
        )
        self.running_mean = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(shape[0], dtype=torch.float), requires_grad=False
                )
                for i, shape in enumerate(f_shapes)
                if i <= self.core.section_num
            ]
        )
        self.beta = beta  # update coefficient for running mean

    def forward(self, *args):
        if len(args) == 1:
            (images,) = args
            logits = self.linear(
                self.core(images)[-1].view(-1, self.linear.weight.shape[1])
            )
            return logits
        if len(args) == 2:
            images_0, images_1 = args
            assert images_0.shape[0] == images_1.shape[0]
            batch_size = images_0.shape[0]
            features_0 = self.core(images_0)
            features_1 = self.core(images_1)

            for i in range(self.core.section_num + 1):
                batch_mean = torch.cat([features_0[i].data, features_1[i].data]).mean(
                    dim=(0, 2, 3)
                )
                self.running_mean[i].data = (
                    self.beta * self.running_mean[i].data + (1 - self.beta) * batch_mean
                )

            scores = torch.stack(
                [
                    F.cosine_similarity(
                        (f_0 - self.running_mean[i][:, None, None]).view(
                            batch_size, -1
                        ),
                        (f_1 - self.running_mean[i][:, None, None]).view(
                            batch_size, -1
                        ),
                    )
                    for i, (f_0, f_1) in enumerate(zip(features_0, features_1))
                    if i <= self.core.section_num
                ],
                dim=1,
            )
            scores = (scores * F.softmax(self.layer_weight, dim=0)).sum(dim=1)
            return scores


def ResNet18(in_shape=(1, 32, 32), num_classes=10, base_channels=64):
    block_nums = [2, 2, 2, 2]
    block_type = "basic"
    return ResNet(in_shape, num_classes, base_channels, block_nums, block_type)


def ResNet34(in_shape=(1, 32, 32), num_classes=10, base_channels=64):
    block_nums = [3, 4, 6, 3]
    block_type = "basic"
    return ResNet(in_shape, num_classes, base_channels, block_nums, block_type)


def ResNet50(in_shape=(1, 32, 32), num_classes=10, base_channels=64):
    block_nums = [3, 4, 6, 3]
    block_type = "bottleneck"
    return ResNet(in_shape, num_classes, base_channels, block_nums, block_type)


def ResNet101(in_shape=(1, 32, 32), num_classes=10, base_channels=64):
    block_nums = [3, 4, 23, 3]
    block_type = "bottleneck"
    return ResNet(in_shape, num_classes, base_channels, block_nums, block_type)
