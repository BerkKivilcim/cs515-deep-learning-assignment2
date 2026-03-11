
"""MobileNetV2 builder used in Part B distillation experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_mobilenet_v2(num_classes: int = 10) -> nn.Module:
    """Build a MobileNetV2 classifier for CIFAR-10.

    Args:
        num_classes: Number of output classes.

    Returns:
        A MobileNetV2 model with a CIFAR-10 classifier head.
    """
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
