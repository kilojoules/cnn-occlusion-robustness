import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
import yaml

# NEW: Import the factory instead of the specific model class
from cnn_occlusion_robustness.models.factory import create_model_from_config


def visualize_filters(
    config_path: str, model_path: str, layer_name: str, output_path: str
):
    """Loads a trained model and visualizes the weights of a specified convolutional layer."""

    # --- 1. Load Configuration and Build the Model using the Factory ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Use the factory to build the model from the 'architecture' list
    architecture = config.get("model", {}).get("architecture")
    if not architecture:
        raise ValueError("Model architecture not found in config file.")

    model = create_model_from_config(architecture)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"Model loaded from {model_path} using architecture from {config_path}")

    # --- 2. Access the Weights of the Specified Layer ---
    try:
        # Dynamically get the layer from the model by its generated name
        layer = dict(model.named_modules())[layer_name]
        if not isinstance(layer, torch.nn.Conv2d):
            raise TypeError(f"Layer '{layer_name}' is not a Conv2d layer.")
    except KeyError:
        print(f"Error: Layer '{layer_name}' not found in the model.")
        print(
            f"Available layers: {[name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Conv2d)]}"
        )
        return

    weights = layer.weight.data

    # Normalize for visualization
    # This logic correctly visualizes filters with 1 or 3 input channels
    if weights.size(1) > 3:
        # For Conv layers with many input channels (e.g., conv2),
        # visualize the filter's response to its first input channel.
        weights = weights[:, 0, :, :].unsqueeze(1)

    # --- 3. Plot the Filters ---
    grid = torchvision.utils.make_grid(weights, nrow=8, padding=1, normalize=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Learned Filters for Layer: '{layer_name}'")
    plt.axis("off")
    plt.savefig("analysis_output/figures/" + output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Filter visualization saved to {output_path}")


def main():
    """Main function to handle command-line arguments and run the visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize the filters of a trained CNN model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config file (.yaml).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.pth).",
    )
    parser.add_argument(
        "--layer-name",
        type=str,
        # UPDATED: The factory names the first Conv2d layer 'conv2d_0'
        default="conv2d_0",
        help="Name of the conv layer to visualize (e.g., 'conv2d_0').",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="filters.png",
        help="Path to save the output image.",
    )
    args = parser.parse_args()

    visualize_filters(args.config, args.model_path, args.layer_name, args.output_path)


if __name__ == "__main__":
    main()
