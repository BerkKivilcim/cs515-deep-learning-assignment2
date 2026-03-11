
"""Evaluation entry points for trained homework models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from parameters import ExperimentConfig
from train import evaluate, get_loaders
from utils import compute_class_accuracy, profile_model, save_json


@torch.no_grad()
def run_test(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    """Load the best checkpoint, evaluate it, and save detailed metrics.

    Args:
        model: Model instance.
        config: Experiment configuration.
        device: Device.
        run_dir: Output directory.

    Returns:
        Dictionary with evaluation statistics.
    """
    _, test_loader = get_loaders(config)
    checkpoint_path = run_dir / "best_model.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss, accuracy = evaluate(model, test_loader, criterion, device)

    all_predictions = []
    all_targets = []
    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    predictions_tensor = torch.cat(all_predictions)
    targets_tensor = torch.cat(all_targets)

    results = {
        "test_loss": loss,
        "test_accuracy": accuracy,
        "profile": profile_model(model, image_size=config.image_size),
    }
    results.update(
        compute_class_accuracy(
            predictions=predictions_tensor,
            targets=targets_tensor,
            num_classes=config.num_classes,
        )
    )

    save_json(results, run_dir / "test_results.json")

    print("\n=== Test Results ===")
    print(f"test_loss={loss:.4f}")
    print(f"test_accuracy={accuracy:.4f}")
    for class_index in range(config.num_classes):
        print(f"class_{class_index}_accuracy={results[f'class_{class_index}_accuracy']:.4f}")

    return results
