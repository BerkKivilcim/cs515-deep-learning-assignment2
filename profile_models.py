from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from ptflops import get_model_complexity_info

from models.CNN import SimpleCNN
from models.ResNet import build_resnet18_scratch
from models.MobileNet import build_mobilenet_v2


def build_model(model_name: str, num_classes: int = 10) -> torch.nn.Module:
    if model_name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if model_name == "resnet18":
        return build_resnet18_scratch(num_classes=num_classes)
    if model_name == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_model(model_name: str, image_size: int = 32) -> Dict[str, str]:
    model = build_model(model_name)
    model.eval()

    macs, params = get_model_complexity_info(
        model,
        (3, image_size, image_size),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )

    total_params = count_params(model)

    return {
        "model": model_name,
        "input_size": f"3x{image_size}x{image_size}",
        "macs": macs,
        "params_ptflops": params,
        "params_exact": str(total_params),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile FLOPs/MACs and parameters for a model.")
    parser.add_argument("--model", choices=["simplecnn", "resnet18", "mobilenet_v2"], required=True)
    parser.add_argument("--image_size", type=int, default=32)
    args = parser.parse_args()

    result = profile_model(args.model, args.image_size)

    print("\n=== Model Complexity Report ===")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()