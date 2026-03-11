
"""Argument parsing and experiment dataclasses for the homework repository."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Container for all experiment-level hyperparameters and I/O paths.

    Attributes:
        mode: Whether to train, test, or do both in a single run.
        task: High-level homework task selector.
        dataset: Dataset name. This homework uses CIFAR-10 for all required tasks.
        model: Model architecture identifier for generic training.
        teacher_model: Teacher model architecture for distillation experiments.
        student_model: Student model architecture for distillation experiments.
        transfer_option: Part A option selector.
        data_dir: Dataset root directory.
        output_dir: Directory where checkpoints and metrics will be stored.
        run_name: Unique experiment name under output_dir.
        device: Preferred device string such as "cuda", "cpu", or "mps".
        seed: Random seed for reproducibility.
        num_workers: Number of DataLoader worker processes.
        batch_size: Batch size.
        test_batch_size: Evaluation batch size.
        epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay coefficient.
        momentum: SGD momentum.
        optimizer: Optimizer name.
        scheduler: Scheduler name.
        step_size: StepLR step size.
        gamma: StepLR gamma value.
        label_smoothing: Label smoothing amount for supervised CE.
        kd_alpha: Interpolation coefficient between hard and soft losses.
        kd_temperature: Softmax temperature for standard knowledge distillation.
        freeze_backbone: Whether to freeze the feature extractor in transfer learning.
        image_size: Target image size after preprocessing.
        save_every: Save an epoch checkpoint every N epochs. Zero disables it.
        log_interval: Print frequency in number of training iterations.
        num_classes: Number of classes.
        export_arch_graphs: Whether to export Torchviz computation graphs.
        graph_batch_size: Batch size used for graph export.
    """

    mode: str
    task: str
    dataset: str
    model: str
    teacher_model: str
    student_model: str
    transfer_option: str
    data_dir: str
    output_dir: str
    run_name: str
    device: str
    seed: int
    num_workers: int
    batch_size: int
    test_batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    momentum: float
    optimizer: str
    scheduler: str
    step_size: int
    gamma: float
    label_smoothing: float
    kd_alpha: float
    kd_temperature: float
    freeze_backbone: bool
    image_size: int
    save_every: int
    log_interval: int
    num_classes: int
    export_arch_graphs: bool
    graph_batch_size: int


def build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI parser for all homework experiments."""
    parser = argparse.ArgumentParser(
        description="CS515 Homework: Transfer Learning and Knowledge Distillation on CIFAR-10"
    )

    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument(
        "--task",
        choices=[
            "partA",
            "partB_simplecnn",
            "partB_teacher",
            "partB_kd_simplecnn",
            "partB_kd_mobilenet",
        ],
        default="partA",
        help="High-level experiment mode.",
    )
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument(
        "--model",
        choices=["simplecnn", "resnet18", "mobilenet_v2", "vgg16"],
        default="resnet18",
        help="Generic model field used by partA and partB_simplecnn/teacher tasks.",
    )
    parser.add_argument(
        "--teacher_model",
        choices=["resnet18"],
        default="resnet18",
        help="Teacher architecture for distillation tasks.",
    )
    parser.add_argument(
        "--student_model",
        choices=["simplecnn", "mobilenet_v2"],
        default="simplecnn",
        help="Student architecture for distillation tasks.",
    )
    parser.add_argument(
        "--transfer_option",
        choices=["resize_freeze", "modify_finetune"],
        default="resize_freeze",
        help="Part A option selector.",
    )

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--run_name", type=str, default="exp")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument(
        "--scheduler",
        choices=["steplr", "cosine", "none"],
        default="steplr",
    )
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--kd_temperature", type=float, default=4.0)

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the backbone for Part A resize_freeze experiments.",
    )
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument(
        "--export_arch_graphs",
        action="store_true",
        help="Export Torchviz architecture graphs for the instantiated model.",
    )
    parser.add_argument(
        "--graph_batch_size",
        type=int,
        default=1,
        help="Batch size of the synthetic input used when exporting Torchviz graphs.",
    )

    return parser


def get_config() -> ExperimentConfig:
    """Parse CLI arguments and return a typed ExperimentConfig instance."""
    parser = build_parser()
    args = parser.parse_args()
    return ExperimentConfig(**vars(args))


def config_to_dict(config: ExperimentConfig) -> Dict[str, object]:
    """Convert a configuration dataclass into a plain dictionary."""
    return asdict(config)
