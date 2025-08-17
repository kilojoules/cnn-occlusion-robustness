import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
import yaml
import argparse
import os  # Make sure os is imported

# NEW: Import the factory
from cnn_occlusion_robustness.models.factory import create_model_from_config
from cnn_occlusion_robustness.train import get_effect


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
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Configuration and Build the Model using the Factory ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    architecture = config.get("model", {}).get("architecture")
    if not architecture:
        raise ValueError("Model architecture not found in config file.")

    model = create_model_from_config(architecture)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"Model loaded from {model_path} using architecture from {config_path}")

    # --- 2. Register Hooks Dynamically ---
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.ReLU)):
            layer.register_forward_hook(get_activation(name))
            print(f"Registered hook for layer: {name}")

    # --- 3. Prepare the Input Image ---
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    clean_image_tensor = transform(image)

    # --- 4. Apply Occlusion Effect ---
    effect_instance = get_effect(test_effect)
    image_np = (clean_image_tensor.permute(1, 2, 0).contiguous().numpy() * 255).astype(
        "uint8"
    )
    augmented_np = effect_instance(image_np)
    augmented_tensor = torch.from_numpy(augmented_np).permute(2, 0, 1) / 255.0
    final_image_tensor = augmented_tensor.float().unsqueeze(0)

    # --- 5. Perform Forward Pass ---
    with torch.no_grad():
        _ = model(final_image_tensor)
    print(f"Forward pass complete for effect '{test_effect}'. Activations captured.")

    # --- 6. Plot the Activations ---

    # NEW: Extract a clean model name from the file path
    model_name = os.path.basename(model_path).replace("_model.pth", "")

    for name, feature_map in activations.items():
        if feature_map.dim() < 4:
            continue

        feature_map = feature_map[0]

        grid = torchvision.utils.make_grid(
            feature_map.unsqueeze(1),
            nrow=8,
            normalize=True,
            pad_value=1,
        )

        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0))

        # UPDATED: Add the model name and effect to the plot title
        plt.title(
            f"Model: '{model_name}' | Layer: '{name}'\n"
            f"Input Effect: '{test_effect}' | Shape: {list(feature_map.shape)}"
        )

        plt.axis("off")

        save_path = os.path.join(output_dir, f"{name}_activations.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
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

    visualize_activations(
        config_path=args.config,
        model_path=args.model_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        test_effect=args.test_effect,
    )


if __name__ == "__main__":
    main()
