from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_key(d: Dict[str, Any], candidates: List[str]) -> Any:
    for key in candidates:
        if key in d:
            return d[key]
    return None


def normalize_history(data: Any) -> Tuple[List[int], Dict[str, List[float]]]:
    metrics = {
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
    }

    # Case 1: list of dicts
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        epochs = []
        for i, row in enumerate(data, start=1):
            epochs.append(int(row.get("epoch", i)))
            metrics["train_acc"].append(
                row.get("train_acc", row.get("train_accuracy", row.get("accuracy_train")))
            )
            metrics["val_acc"].append(
                row.get("val_acc", row.get("val_accuracy", row.get("valid_acc")))
            )
            metrics["test_acc"].append(
                row.get("test_acc", row.get("test_accuracy", row.get("accuracy_test")))
            )
            metrics["train_loss"].append(
                row.get("train_loss", row.get("loss_train"))
            )
            metrics["val_loss"].append(
                row.get("val_loss", row.get("valid_loss"))
            )
            metrics["test_loss"].append(
                row.get("test_loss", row.get("loss_test"))
            )
        return epochs, metrics

    # Case 2: dict with history
    if isinstance(data, dict) and "history" in data:
        return normalize_history(data["history"])

    # Case 3: dict of lists
    if isinstance(data, dict):
        epochs = find_key(data, ["epochs", "epoch"])
        if epochs is None:
            candidate = find_key(
                data,
                [
                    "train_acc", "train_accuracy", "test_acc", "test_accuracy",
                    "train_loss", "val_loss", "test_loss",
                    "train_accs", "val_accs", "test_accs",
                    "train_losses", "val_losses", "test_losses",
                ],
            )
            if candidate is None:
                raise ValueError(
                    f"Unsupported metrics.json structure. Top-level keys: {list(data.keys())}"
                )
            epochs = list(range(1, len(candidate) + 1))

        metrics["train_acc"] = find_key(
            data, ["train_acc", "train_accuracy", "accuracy_train", "train_accs"]
        ) or []
        metrics["val_acc"] = find_key(
            data, ["val_acc", "val_accuracy", "valid_acc", "val_accs"]
        ) or []
        metrics["test_acc"] = find_key(
            data, ["test_acc", "test_accuracy", "accuracy_test", "test_accs"]
        ) or []
        metrics["train_loss"] = find_key(
            data, ["train_loss", "loss_train", "train_losses"]
        ) or []
        metrics["val_loss"] = find_key(
            data, ["val_loss", "valid_loss", "val_losses"]
        ) or []
        metrics["test_loss"] = find_key(
            data, ["test_loss", "loss_test", "test_losses"]
        ) or []

        return list(epochs), metrics

    raise ValueError(f"Unsupported metrics.json format. Type: {type(data)}")


def clean(values: List[Any]) -> List[float]:
    cleaned = []
    for v in values:
        cleaned.append(float("nan") if v is None else float(v))
    return cleaned


def plot_single_run(metrics_path: Path, output_dir: Path) -> None:
    data = load_json(metrics_path)
    epochs, metrics = normalize_history(data)

    run_name = metrics_path.parent.name
    print(f"[INFO] Plotting: {metrics_path}")

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plotted = False
    for key, label in [
        ("train_acc", "Train Accuracy"),
        ("val_acc", "Validation Accuracy"),
        ("test_acc", "Test Accuracy"),
    ]:
        if len(metrics[key]) > 0:
            plt.plot(
                epochs[: len(metrics[key])],
                clean(metrics[key]),
                marker="o",
                label=label,
            )
            plotted = True

    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curves - {run_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = output_dir / f"{run_name}_accuracy.png"
        plt.savefig(save_path, dpi=200)
        print(f"[SAVED] {save_path}")
    else:
        print(f"[WARN] No accuracy fields found in {metrics_path}")
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plotted = False
    for key, label in [
        ("train_loss", "Train Loss"),
        ("val_loss", "Validation Loss"),
        ("test_loss", "Test Loss"),
    ]:
        if len(metrics[key]) > 0:
            plt.plot(
                epochs[: len(metrics[key])],
                clean(metrics[key]),
                marker="o",
                label=label,
            )
            plotted = True

    if plotted:
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves - {run_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = output_dir / f"{run_name}_loss.png"
        plt.savefig(save_path, dpi=200)
        print(f"[SAVED] {save_path}")
    else:
        print(f"[WARN] No loss fields found in {metrics_path}")
    plt.close()


def main() -> None:
    root = Path("outputs")
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Current working directory: {Path.cwd()}")
    print(f"[INFO] Looking for metrics.json files under: {root.resolve()}")

    metrics_files = list(root.rglob("metrics.json"))

    print(f"[INFO] Found {len(metrics_files)} metrics.json files.")
    for path in metrics_files:
        print(f"  - {path}")

    if not metrics_files:
        print("[ERROR] No metrics.json files found.")
        return

    for metrics_path in metrics_files:
        try:
            plot_single_run(metrics_path, output_dir)
        except Exception as e:
            print(f"[ERROR] Skipped {metrics_path}: {e}")

    print(f"[DONE] All plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()