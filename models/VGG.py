
"""VGG builders for transfer-learning baselines."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights


def build_vgg16_resize_freeze(
    num_classes: int = 10,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a pretrained VGG16 model for resize-and-freeze transfer learning.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze the feature extractor.

    Returns:
        A VGG16 model adapted to CIFAR-10.
    """
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
