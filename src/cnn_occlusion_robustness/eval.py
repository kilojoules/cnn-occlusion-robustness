import argparse
import json
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

from cnn_occlusion_robustness.data.gtsrb import GTSRBDataset
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN
from cnn_occlusion_robustness.train import get_effect, AugmentedDataset


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_from_config(
    config_path: str, weights_path: str, device: torch.device
) -> SimpleCNN:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_params = (cfg or {}).get("model", {}).get("params", {})
    if not model_params:
        raise ValueError(
            "Model parameters not found in config['model']['params']. "
            "Ensure your YAML includes the 'model: { params: ... }' block."
        )

    model = SimpleCNN(**model_params).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    # Use strict=True to fail fast if the architecture doesn't match.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_model(
    config_path: str,
    model_path: str,
    data_dir: str,
    test_effect: str,
    output_path: str,
    batch_size: int = 128,
    num_workers: Optional[int] = None,
):
    """Evaluates a single trained model on a single test condition."""

    device = _resolve_device()
    print(f"Using device: {device}")

    # 1) Load model with the exact train-time architecture
    model = _load_model_from_config(config_path, model_path, device)
    print(f"Loaded model from '{model_path}' using architecture in '{config_path}'")

    # 2) Prepare dataset (+ optional occlusion augmentation for eval)
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Test data directory not found: {data_dir}\n"
            "Make sure you've run scripts/organize_test_set.py for the test set."
        )

    full_dataset = GTSRBDataset(root_dir=data_dir, transform=transform)
    effect = get_effect(test_effect)
    augmented_test_dataset = AugmentedDataset(full_dataset, effect)

    if num_workers is None:
        # A small default that tends to work cross-platform
        num_workers = 2

    test_loader = DataLoader(
        augmented_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 3) Evaluate
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating on '{test_effect}'"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy on '{test_effect}': {accuracy:.4f}")

    # 4) Save result JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "config_path": config_path,
        "model_path": model_path,
        "test_effect": test_effect,
        "data_dir": data_dir,
        "batch_size": batch_size,
        "accuracy": accuracy,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN model on a specified test condition."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config used to define the model architecture.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model weights (.pth).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the organized GTSRB test dataset (class subfolders).",
    )
    parser.add_argument(
        "--test-effect",
        type=str,
        required=True,
        help="Occlusion effect to apply during testing (e.g., 'none', 'heavy_rain').",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the evaluation result JSON.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader workers (default: 2)."
    )
    args = parser.parse_args()

    evaluate_model(
        config_path=args.config,
        model_path=args.model_path,
        data_dir=args.data_dir,
        test_effect=args.test_effect,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
