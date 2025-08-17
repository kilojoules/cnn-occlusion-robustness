import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def _smooth(y: list[float], k: int) -> np.ndarray:
    """Applies a simple moving average filter."""
    y_arr = np.asarray(y, dtype=float)
    if k <= 1 or len(y_arr) < k:
        return y_arr
    # Use valid mode to avoid edge effects, then pad to original length
    smoothed = np.convolve(y_arr, np.ones(k) / k, mode="valid")
    # Pad at the beginning to align the smoothed curve
    pad_width = len(y_arr) - len(smoothed)
    return np.pad(smoothed, (pad_width, 0), "edge")


def plot_curves_for_file(json_path: str, output_dir: str, smooth_k: int = 1):
    """
    Creates and saves a single plot with training/validation loss and accuracy
    curves from a single training results JSON file.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Results JSON not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Safely extract history data
    hist = (data or {}).get("history", {})
    train_loss = hist.get("train_loss", [])
    val_loss = hist.get("val_loss", [])
    train_acc = hist.get("train_acc", [])
    val_acc = hist.get("val_acc", [])

    if not train_loss:
        raise ValueError("No training history found in JSON file.")

    n = len(train_loss)
    epochs = np.arange(1, n + 1)

    # --- Create a single figure with two subplots ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 8), sharex=True, constrained_layout=True
    )

    # --- Plot Loss on the first subplot (ax1) ---
    ax1.plot(epochs, _smooth(train_loss, smooth_k), "o-", label="Train Loss", alpha=0.8)
    if val_loss:
        ax1.plot(
            epochs,
            _smooth(val_loss, smooth_k),
            "o-",
            label="Validation Loss",
            alpha=0.8,
        )
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Over Epochs")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.legend()

    # --- Plot Accuracy on the second subplot (ax2) ---
    if train_acc or val_acc:
        if train_acc:
            ax2.plot(
                epochs,
                _smooth(train_acc, smooth_k),
                "o-",
                label="Train Accuracy",
                alpha=0.8,
            )
        if val_acc:
            ax2.plot(
                epochs,
                _smooth(val_acc, smooth_k),
                "o-",
                label="Validation Accuracy",
                alpha=0.8,
            )
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)  # Set a consistent y-axis for accuracy
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax2.legend()

    ax2.set_xlabel("Epoch")

    # --- Final Touches ---
    base_name = (
        os.path.basename(json_path)
        .replace("_model_results.json", "")
        .replace(".json", "")
    )
    fig.suptitle(f"Training Curves: '{base_name}'", fontsize=16, fontweight="bold")

    # Save the combined figure
    output_filename = f"{base_name}_training_curves.png"
    output_path = os.path.join(output_dir, output_filename)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """
    Finds all training result JSON files in a directory and generates a
    combined plot for each one.
    """
    parser = argparse.ArgumentParser(
        description="Plot training/validation curves for all models in a results directory."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing the training output files (e.g., 'results/models/').",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output/figures/training_curves",
        help="Directory where the plot images will be saved.",
    )
    parser.add_argument(
        "--smooth-k",
        type=int,
        default=1,
        help="Window size for simple moving average smoothing (1 = no smoothing).",
    )
    args = parser.parse_args()

    # Find all relevant JSON files
    json_files = glob.glob(os.path.join(args.results_dir, "*_results.json"))

    if not json_files:
        print(f"Error: No '*_results.json' files found in '{args.results_dir}'.")
        print("Please ensure you have run the training script first.")
        return

    print(f"Found {len(json_files)} training result file(s). Generating plots...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each file
    for json_path in tqdm(json_files, desc="Generating Plots"):
        plot_curves_for_file(
            json_path=json_path,
            output_dir=args.output_dir,
            smooth_k=args.smooth_k,
        )

    print(f"\n✅ Plots saved successfully to '{args.output_dir}'")


if __name__ == "__main__":
    main()
