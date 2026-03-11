
"""Small CNN baselines used for CIFAR-10 experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple convolutional neural network for CIFAR-10 classification.

    Architecture:
        Conv(3→32) -> ReLU -> MaxPool
        Conv(32→64) -> ReLU -> MaxPool
        Conv(64→128) -> ReLU -> MaxPool
        Flatten -> Linear(2048→256) -> ReLU -> Dropout -> Linear(256→num_classes)

    Args:
        num_classes: Number of target classes.
        dropout: Dropout probability before the final classifier.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize Conv2d and Linear layers with Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return class logits."""
        x = self.features(x)
        x = self.classifier(x)
        return x
