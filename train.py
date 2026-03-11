
"""Training and data-loading utilities for all homework experiments."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from losses import (
    LabelSmoothingCrossEntropy,
    custom_kd_loss,
    kd_loss,
)
from parameters import ExperimentConfig
from utils import profile_model, save_json


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_transforms(config: ExperimentConfig, train: bool) -> transforms.Compose:
    """Build preprocessing and augmentation transforms.

    Args:
        config: Experiment configuration.
        train: Whether the transform is for training mode.

    Returns:
        torchvision Compose object.
    """
    transform_list = []
    if config.image_size != 32:
        transform_list.append(transforms.Resize((config.image_size, config.image_size)))

    if train:
        if config.image_size == 32:
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
            ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(transform_list)


def get_loaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test loaders.

    Args:
        config: Experiment configuration.

    Returns:
        Tuple of train_loader and test_loader.
    """
    train_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=get_transforms(config, train=True),
    )
    test_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=get_transforms(config, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_optimizer(config: ExperimentConfig, model: nn.Module) -> Optimizer:
    """Build the optimizer using only trainable parameters."""
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == "adam":
        return torch.optim.Adam(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        trainable_parameters,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def build_scheduler(config: ExperimentConfig, optimizer: Optimizer):
    """Build the requested learning rate scheduler."""
    if config.scheduler == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
    return None


def supervised_criterion(config: ExperimentConfig) -> nn.Module:
    """Build the supervised loss criterion."""
    return LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)


def train_one_epoch_supervised(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> Tuple[float, float]:
    """Train a model for one supervised epoch.

    Args:
        model: Model to optimize.
        loader: Training data loader.
        optimizer: Optimizer.
        criterion: Supervised criterion.
        device: Device.
        log_interval: Logging interval in steps.

    Returns:
        Tuple of epoch_loss and epoch_accuracy.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_samples += batch_size

        if step % log_interval == 0:
            print(
                f"  [step {step:04d}/{len(loader):04d}] "
                f"loss={total_loss / total_samples:.4f} "
                f"acc={total_correct / total_samples:.4f}"
            )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate a model on a data loader.

    Args:
        model: Model to evaluate.
        loader: Evaluation loader.
        criterion: Loss function.
        device: Device.

    Returns:
        Tuple of loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def run_supervised_training(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    """Run standard supervised training and save the best checkpoint.

    Args:
        model: Model to train.
        config: Experiment configuration.
        device: Device.
        run_dir: Output directory for this run.

    Returns:
        Dictionary containing training summary information.
    """
    train_loader, test_loader = get_loaders(config)
    criterion = supervised_criterion(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    best_acc = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss, train_acc = train_one_epoch_supervised(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_interval=config.log_interval,
        )
        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} lr={current_lr:.6f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr,
        }
        history.append(epoch_record)

        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, run_dir / "best_model.pth")
            print(f"  Saved best checkpoint with test_acc={best_acc:.4f}")

        if config.save_every > 0 and epoch % config.save_every == 0:
            torch.save(model.state_dict(), run_dir / f"epoch_{epoch:03d}.pth")

    model.load_state_dict(best_weights)

    summary = {
        "best_test_accuracy": best_acc,
        "history": history,
        "profile": profile_model(model, image_size=config.image_size),
    }
    save_json(summary, run_dir / "metrics.json")
    return summary


def train_one_epoch_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    config: ExperimentConfig,
    kd_variant: str,
) -> Tuple[float, float]:
    """Train one epoch with knowledge distillation.

    Args:
        student: Student model to optimize.
        teacher: Teacher model in eval mode.
        loader: Training data loader.
        optimizer: Optimizer for the student.
        device: Device.
        config: Experiment configuration.
        kd_variant: Either "standard" or "custom_true_class".

    Returns:
        Tuple of epoch loss and epoch accuracy.
    """
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)

        if kd_variant == "custom_true_class":
            loss = custom_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=targets,
                alpha=config.kd_alpha,
            )
        else:
            loss = kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                targets=targets,
                alpha=config.kd_alpha,
                temperature=config.kd_temperature,
            )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += student_logits.argmax(dim=1).eq(targets).sum().item()
        total_samples += batch_size

        if step % config.log_interval == 0:
            print(
                f"  [step {step:04d}/{len(loader):04d}] "
                f"loss={total_loss / total_samples:.4f} "
                f"acc={total_correct / total_samples:.4f}"
            )

    return total_loss / total_samples, total_correct / total_samples


def run_distillation_training(
    student: nn.Module,
    teacher: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    run_dir: Path,
    kd_variant: str = "standard",
) -> Dict[str, object]:
    """Run a distillation experiment and save the best student checkpoint.

    Args:
        student: Student model to train.
        teacher: Frozen teacher model.
        config: Experiment configuration.
        device: Device.
        run_dir: Output directory.
        kd_variant: Distillation variant name.

    Returns:
        Dictionary containing training summary information.
    """
    train_loader, test_loader = get_loaders(config)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, student)
    scheduler = build_scheduler(config, optimizer)

    best_acc = -1.0
    best_weights = copy.deepcopy(student.state_dict())
    history = []

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss, train_acc = train_one_epoch_kd(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            kd_variant=kd_variant,
        )
        test_loss, test_acc = evaluate(
            model=student,
            loader=test_loader,
            criterion=eval_criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} lr={current_lr:.6f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr,
        }
        history.append(epoch_record)

        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = copy.deepcopy(student.state_dict())
            torch.save(best_weights, run_dir / "best_model.pth")
            print(f"  Saved best student checkpoint with test_acc={best_acc:.4f}")

    student.load_state_dict(best_weights)

    summary = {
        "best_test_accuracy": best_acc,
        "history": history,
        "student_profile": profile_model(student, image_size=config.image_size),
        "teacher_profile": profile_model(teacher, image_size=config.image_size),
        "kd_variant": kd_variant,
    }
    save_json(summary, run_dir / "metrics.json")
    return summary
