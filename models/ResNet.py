"""ResNet model helpers for CIFAR-10 and transfer learning experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18_pretrained_original():
    """
    Load the original pretrained ResNet18 trained on ImageNet.
    This version is used only for architecture visualization
    before modifying the network for CIFAR10.
    """

    from torchvision import models
    from torchvision.models import ResNet18_Weights

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    return model


def build_resnet18_scratch(num_classes: int = 10) -> nn.Module:
    """Build a CIFAR-friendly ResNet-18 from scratch.

    This version keeps the standard torchvision ResNet-18 body but modifies the
    early stem to better match 32×32 CIFAR-10 inputs.

    Args:
        num_classes: Number of output classes.

    Returns:
        A torchvision ResNet-18 model initialized from scratch.
    """
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_transfer_resnet18_resize_freeze(
    num_classes: int = 10,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build ImageNet-pretrained ResNet-18 for the resize-and-freeze strategy.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze all parameters except the final FC layer.

    Returns:
        A pretrained ResNet-18 adapted for CIFAR-10.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def build_transfer_resnet18_modify_finetune(num_classes: int = 10) -> nn.Module:
    """Build a modified pretrained ResNet-18 for CIFAR-10 fine-tuning.

    The function starts from ImageNet pretrained weights, replaces the early
    7×7 / stride-2 convolution with a 3×3 / stride-1 convolution, removes the
    max-pooling layer, copies the center 3×3 part of pretrained filters into
    the new convolution, and fine-tunes all layers.

    Args:
        num_classes: Number of output classes.

    Returns:
        A modified pretrained ResNet-18 adapted to 32×32 inputs.
    """
    pretrained = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model = models.resnet18(weights=None)

    new_conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    with torch.no_grad():
        new_conv1.weight.copy_(pretrained.conv1.weight[:, :, 2:5, 2:5])

    model.conv1 = new_conv1
    model.maxpool = nn.Identity()

    state_dict = pretrained.state_dict()
    state_dict.pop("conv1.weight", None)
    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)

    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for parameter in model.parameters():
        parameter.requires_grad = True

    return model