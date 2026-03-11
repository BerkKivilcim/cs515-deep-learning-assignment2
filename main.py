"""Main entry point for all homework experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from parameters import ExperimentConfig, config_to_dict, get_config
from models.CNN import SimpleCNN
from models.MobileNet import build_mobilenet_v2
from models.ResNet import (
    build_resnet18_pretrained_original,
    build_resnet18_scratch,
    build_transfer_resnet18_modify_finetune,
    build_transfer_resnet18_resize_freeze,
)
from models.VGG import build_vgg16_resize_freeze
from test import run_test
from train import run_distillation_training, run_supervised_training
from utils import export_torchviz_graph, prepare_run_dir, resolve_device, save_json, set_seed


def build_generic_model(config: ExperimentConfig) -> nn.Module:
    """Build a model according to the generic model selector.

    Args:
        config: Experiment configuration.

    Returns:
        Instantiated model.
    """
    if config.model == "simplecnn":
        return SimpleCNN(num_classes=config.num_classes)
    if config.model == "resnet18":
        return build_resnet18_scratch(num_classes=config.num_classes)
    if config.model == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=config.num_classes)
    if config.model == "vgg16":
        return build_vgg16_resize_freeze(
            num_classes=config.num_classes,
            freeze_backbone=False,
        )
    raise ValueError(f"Unsupported model: {config.model}")


def build_teacher_model(config: ExperimentConfig) -> nn.Module:
    """Build the teacher architecture for Part B distillation."""
    if config.teacher_model == "resnet18":
        return build_resnet18_scratch(num_classes=config.num_classes)
    raise ValueError(f"Unsupported teacher model: {config.teacher_model}")


def build_student_model(config: ExperimentConfig) -> nn.Module:
    """Build the student architecture for Part B distillation."""
    if config.student_model == "simplecnn":
        return SimpleCNN(num_classes=config.num_classes)
    if config.student_model == "mobilenet_v2":
        return build_mobilenet_v2(num_classes=config.num_classes)
    raise ValueError(f"Unsupported student model: {config.student_model}")


def export_graph_with_metadata(
    model: nn.Module,
    run_dir: Path,
    graph_name: str,
    device: torch.device,
    image_size: int,
    graph_batch_size: int,
) -> None:
    """Export a Torchviz graph plus a small JSON metadata file.

    Args:
        model: Model to visualize.
        run_dir: Output directory of the current experiment.
        graph_name: File stem to use for the exported graph.
        device: Device used for the synthetic forward pass.
        image_size: Spatial size of the dummy image.
        graph_batch_size: Dummy batch size used for graph export.
    """
    graph_info = export_torchviz_graph(
        model=model,
        output_stem=run_dir / graph_name,
        image_size=image_size,
        batch_size=graph_batch_size,
        device=device,
    )
    save_json(graph_info, run_dir / f"{graph_name}_graph.json")
    if graph_info.get("status") == "ok":
        print(f"Torchviz graph exported to: {graph_info['png_path']}")
    else:
        print(f"Torchviz graph export skipped: {graph_info}")


def export_part_a_resnet_graphs(config: ExperimentConfig, run_dir: Path, device: torch.device, adapted_model: nn.Module) -> None:
    """Export both original and adapted Part A ResNet-18 graphs.

    For Part A with ResNet-18, the report often benefits from showing two
    separate graphs: the original ImageNet-pretrained architecture and the
    CIFAR-10-adapted architecture actually used in the experiment.

    Args:
        config: Experiment configuration.
        run_dir: Output directory of the current experiment.
        device: Device used for graph export.
        adapted_model: The already-built CIFAR-10-adapted model.
    """
    if not config.export_arch_graphs:
        return

    original_model = build_resnet18_pretrained_original().to(device)
    export_graph_with_metadata(
        model=original_model,
        run_dir=run_dir,
        graph_name="original_pretrained_architecture",
        device=device,
        image_size=224,
        graph_batch_size=config.graph_batch_size,
    )

    adapted_image_size = 224 if config.transfer_option == "resize_freeze" else 32
    export_graph_with_metadata(
        model=adapted_model,
        run_dir=run_dir,
        graph_name="adapted_architecture",
        device=device,
        image_size=adapted_image_size,
        graph_batch_size=config.graph_batch_size,
    )


def maybe_export_graph(
    model: nn.Module,
    config: ExperimentConfig,
    run_dir: Path,
    graph_name: str,
    image_size: Optional[int] = None,
) -> None:
    """Optionally export a Torchviz architecture graph for a single model."""
    if not config.export_arch_graphs:
        return

    export_graph_with_metadata(
        model=model,
        run_dir=run_dir,
        graph_name=graph_name,
        device=resolve_device(config.device),
        image_size=config.image_size if image_size is None else image_size,
        graph_batch_size=config.graph_batch_size,
    )


def build_part_a_model(config: ExperimentConfig) -> nn.Module:
    """Build the model for Part A transfer-learning experiments.

    Args:
        config: Experiment configuration.

    Returns:
        Transfer-learning model.
    """
    if config.model == "resnet18":
        if config.transfer_option == "resize_freeze":
            return build_transfer_resnet18_resize_freeze(
                num_classes=config.num_classes,
                freeze_backbone=config.freeze_backbone,
            )
        return build_transfer_resnet18_modify_finetune(
            num_classes=config.num_classes
        )

    if config.model == "vgg16":
        if config.transfer_option != "resize_freeze":
            raise ValueError("VGG16 is only wired for the resize_freeze Part A option.")
        return build_vgg16_resize_freeze(
            num_classes=config.num_classes,
            freeze_backbone=config.freeze_backbone,
        )

    raise ValueError("Part A currently supports resnet18 or vgg16.")


def main() -> None:
    """Parse arguments, build the requested experiment, and run it."""
    config = get_config()
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(config.output_dir, config.run_name)

    save_json(config_to_dict(config), run_dir / "config.json")

    print(f"Running task={config.task} on device={device}")
    print(f"Outputs will be stored in: {run_dir}")

    if config.task == "partA":
        model = build_part_a_model(config).to(device)
        if config.model == "resnet18":
            export_part_a_resnet_graphs(config, run_dir, device, model)
        else:
            maybe_export_graph(model, config, run_dir, graph_name="model_architecture")
        if config.mode in ("train", "both"):
            run_supervised_training(model=model, config=config, device=device, run_dir=run_dir)
        if config.mode in ("test", "both"):
            run_test(model=model, config=config, device=device, run_dir=run_dir)
        return

    if config.task in ("partB_simplecnn", "partB_teacher"):
        model = build_generic_model(config).to(device)
        maybe_export_graph(model, config, run_dir, graph_name="model_architecture")
        if config.mode in ("train", "both"):
            run_supervised_training(model=model, config=config, device=device, run_dir=run_dir)
        if config.mode in ("test", "both"):
            run_test(model=model, config=config, device=device, run_dir=run_dir)
        return

    if config.task in ("partB_kd_simplecnn", "partB_kd_mobilenet"):
        teacher = build_teacher_model(config).to(device)
        maybe_export_graph(teacher, config, run_dir, graph_name="teacher_architecture")

        teacher_checkpoint = Path(config.output_dir) / config.teacher_model / "best_model.pth"
        if not teacher_checkpoint.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher_checkpoint}. "
                "Train the teacher first and store it under output_dir/teacher_model/best_model.pth "
                "or point output_dir accordingly."
            )
        teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

        student = build_student_model(config).to(device)
        maybe_export_graph(student, config, run_dir, graph_name="student_architecture")
        kd_variant = "standard" if config.task == "partB_kd_simplecnn" else "custom_true_class"

        if config.mode in ("train", "both"):
            run_distillation_training(
                student=student,
                teacher=teacher,
                config=config,
                device=device,
                run_dir=run_dir,
                kd_variant=kd_variant,
            )
        if config.mode in ("test", "both"):
            run_test(model=student, config=config, device=device, run_dir=run_dir)
        return

    raise ValueError(f"Unsupported task: {config.task}")


if __name__ == "__main__":
    main()