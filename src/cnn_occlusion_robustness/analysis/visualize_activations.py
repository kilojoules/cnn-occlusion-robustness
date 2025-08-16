import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
import yaml
import argparse
from cnn_occlusion_robustness.train import get_effect
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN

# Dictionary to store the activations
activations = {}


def get_activation(name):
    """Hook function to capture the output of a layer."""

    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


def visualize_activations(
    config_path: str,
    model_path: str,
    image_path: str,
    output_dir: str,
    test_effect: str,
):
    """Loads a model and an image, and visualizes the layer activations."""

    # --- 1. Load Configuration and Build the Model ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_params = config.get("model", {}).get("params", {})
    if not model_params:
        raise ValueError("Model parameters not found in config file.")

    # Build the model with the correct architecture
    model = SimpleCNN(**model_params)

    # Load the trained weights into the correctly-structured model
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"Model loaded from {model_path} using architecture from {config_path}")

    # --- 2. Register Hooks ---
    # This part remains the same. It correctly targets the hidden layers.
    model.conv1.register_forward_hook(get_activation("conv1"))
    model.pool.register_forward_hook(get_activation("pool1"))
    model.conv2.register_forward_hook(get_activation("conv2"))

    # --- 3. Prepare the Input Image ---
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    clean_image_tensor = transform(image)  # Start with a clean tensor

    # --- 4. APPLY OCCLUSION EFFECT (NEW STEP) ---
    effect_instance = get_effect(test_effect)

    # Convert tensor to numpy, apply effect, and convert back to tensor
    # This logic is borrowed from the AugmentedDataset in train.py
    image_np = (clean_image_tensor.permute(1, 2, 0).contiguous().numpy() * 255).astype(
        "uint8"
    )
    augmented_np = effect_instance(image_np)
    augmented_tensor = torch.from_numpy(augmented_np).permute(2, 0, 1) / 255.0

    # Add the batch dimension for the model
    final_image_tensor = augmented_tensor.float().unsqueeze(0)

    # --- 5. Perform Forward Pass ---
    with torch.no_grad():
        # Use the potentially augmented tensor
        _ = model(final_image_tensor)

    print(f"Forward pass complete for effect '{test_effect}'. Activations captured.")

    # --- 6. Plot the Activations ---
    for name, feature_map in activations.items():
        # The feature map is a 4D tensor (batch, channels, height, width)
        # We take the first item in the batch
        feature_map = feature_map[0]

        # Make a grid of the channels
        grid = torchvision.utils.make_grid(
            feature_map.unsqueeze(1),  # Add a dimension for grayscale
            nrow=4,  # Adjust number of columns in the grid
            normalize=True,
            pad_value=1,
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"Activations for Layer: '{name}'\nShape: {list(feature_map.shape)}")
        plt.axis("off")

        save_path = f"{output_dir}/{name}_activations.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Activation plot for '{name}' saved to {save_path}")


def main():
    """Main function to handle command-line arguments and run the visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize layer activations for a given image."
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
        "--image-path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="activations_output",
        help="Directory to save output images.",
    )
    parser.add_argument(
        "--test-effect",
        type=str,
        default="none",
        help="Occlusion effect to apply to the input image.",
    )

    args = parser.parse_args()

    import os

    os.makedirs(args.output_dir, exist_ok=True)

    visualize_activations(
        config_path=args.config,
        model_path=args.model_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        test_effect=args.test_effect,
    )


if __name__ == "__main__":
    main()
