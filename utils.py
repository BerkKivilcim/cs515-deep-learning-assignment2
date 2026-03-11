"""Utility functions for reproducibility, I/O, metrics, profiling, and graph export."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

try:
    from ptflops import get_model_complexity_info
except Exception:  # pragma: no cover - optional runtime dependency
    get_model_complexity_info = None

try:
    from torchviz import make_dot
except Exception:  # pragma: no cover - optional runtime dependency
    make_dot = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(preferred_device: str) -> torch.device:
    """Resolve the best available device from the user's preferred string."""
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_run_dir(output_dir: str, run_name: str) -> Path:
    """Create and return the output directory for a specific experiment run."""
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(payload: Dict[str, object], path: Path) -> None:
    """Save a dictionary as a JSON file."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def compute_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Compute per-class accuracy from predictions and targets."""
    result: Dict[str, float] = {}
    for class_index in range(num_classes):
        mask = targets == class_index
        total = int(mask.sum().item())
        correct = int((predictions[mask] == targets[mask]).sum().item()) if total > 0 else 0
        result[f"class_{class_index}_accuracy"] = float(correct / total) if total > 0 else 0.0
    return result


def profile_model(model: torch.nn.Module, image_size: int = 32) -> Dict[str, str]:
    """Profile FLOPs and parameter counts using ptflops.

    Args:
        model: Model to profile.
        image_size: Input spatial size.

    Returns:
        Dictionary containing string-formatted FLOPs and parameter counts.
    """
    if get_model_complexity_info is None:
        return {"macs": "ptflops_not_installed", "params": "ptflops_not_installed"}

    was_training = model.training
    model.eval()
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            (3, image_size, image_size),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
    model.train(was_training)
    return {"macs": macs, "params": params}


def export_torchviz_graph(
    model: torch.nn.Module,
    output_stem: Path,
    image_size: int = 32,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, str]:
    """Export a Torchviz computation graph for a model.

    Args:
        model: Model to visualize.
        output_stem: Output path stem without file suffix.
        image_size: Synthetic input spatial size.
        batch_size: Synthetic input batch size.
        device: Device on which the synthetic forward pass will be executed.

    Returns:
        A dictionary describing whether graph export succeeded and where files were written.
    """
    if make_dot is None:
        return {
            "status": "skipped",
            "reason": "torchviz_not_installed",
        }

    target_device = device if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()

    dummy = torch.randn(batch_size, 3, image_size, image_size, device=target_device)
    output = model(dummy)
    graph = make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )
    graph.format = "png"
    rendered_path = Path(graph.render(str(output_stem), cleanup=False))

    model.train(was_training)
    return {
        "status": "ok",
        "png_path": str(rendered_path),
        "dot_path": str(output_stem.with_suffix('.gv')),
    }