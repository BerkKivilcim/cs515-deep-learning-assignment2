
"""Loss functions for supervised learning and distillation."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Args:
        smoothing: Smoothing amount in [0, 1).
    """

    def __init__(self, smoothing: float = 0.0) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy."""
        if self.smoothing <= 0.0:
            return F.cross_entropy(logits, targets)

        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = torch.sum(-true_dist * log_probs, dim=1).mean()
        return loss


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """Compute standard knowledge distillation loss.

    The loss interpolates between hard cross-entropy and soft KL divergence.

    Args:
        student_logits: Student logits of shape (B, C).
        teacher_logits: Teacher logits of shape (B, C).
        targets: Integer labels of shape (B,).
        alpha: Weight for the KD term.
        temperature: Softmax temperature.

    Returns:
        Scalar loss tensor.
    """
    hard_loss = F.cross_entropy(student_logits, targets)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    soft_loss = soft_loss * (temperature ** 2)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def custom_teacher_distribution(
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Create the custom Part B target distribution for MobileNet distillation.

    The teacher's probability assigned to the true class is preserved. The
    remaining mass is distributed uniformly over all other classes.

    Args:
        teacher_logits: Teacher logits of shape (B, C).
        targets: Ground-truth class indices of shape (B,).

    Returns:
        A probability tensor of shape (B, C).
    """
    teacher_probs = F.softmax(teacher_logits, dim=1)
    batch_size, num_classes = teacher_probs.shape
    true_class_probs = teacher_probs.gather(1, targets.unsqueeze(1))

    remaining = 1.0 - true_class_probs
    off_value = remaining / (num_classes - 1)

    custom_probs = torch.full_like(teacher_probs, 0.0)
    custom_probs += off_value
    custom_probs.scatter_(1, targets.unsqueeze(1), true_class_probs)
    return custom_probs


def custom_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute the custom distillation loss for the MobileNet student.

    Args:
        student_logits: Student logits of shape (B, C).
        teacher_logits: Teacher logits of shape (B, C).
        targets: Ground-truth labels of shape (B,).
        alpha: Weight for the custom KD term.

    Returns:
        Scalar loss tensor.
    """
    hard_loss = F.cross_entropy(student_logits, targets)
    target_probs = custom_teacher_distribution(teacher_logits, targets)
    student_log_probs = F.log_softmax(student_logits, dim=1)
    soft_loss = F.kl_div(student_log_probs, target_probs, reduction="batchmean")
    return alpha * soft_loss + (1.0 - alpha) * hard_loss
