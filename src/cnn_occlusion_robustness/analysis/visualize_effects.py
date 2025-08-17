import argparse
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import the get_effect function directly from your project's train script
from cnn_occlusion_robustness.train import get_effect


def visualize_effects(effects: list[str], image_path: str, output_path: str, cols: int):
    """
    Generates and saves a grid of images showing the specified occlusion effects.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Base image not found at: {image_path}")

    base_img = Image.open(image_path).convert("RGB").resize((128, 128))

    # Calculate grid size
    num_effects = len(effects)
    rows = math.ceil(num_effects / cols)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True
    )
    # Ensure axes is always a 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    print(f"Generating {num_effects} effect visualizations...")
    for i, effect_name in enumerate(tqdm(effects, desc="Applying effects")):
        ax = axes[i]

        # Get the effect instance using the project's factory function
        effect_instance = get_effect(effect_name)

        # Convert PIL image to numpy array for the effect function
        img_np = np.array(base_img)

        # Apply the effect
        augmented_np = effect_instance(img_np)

        ax.imshow(augmented_np)
        ax.set_title(effect_name.replace("_", " ").title(), fontsize=12)
        ax.axis("off")

    # Hide any unused subplots
    for i in range(num_effects, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Occlusion Effect Gallery", fontsize=16, fontweight="bold")

    # Save the final figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Effect gallery saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a gallery of specified occlusion effects."
    )
    parser.add_argument(
        "--effects",
        nargs="+",
        required=True,
        help="A list of effect names to visualize (e.g., 'none' 'light_rain').",
    )
    parser.add_argument(
        "--image-path",
        default="assets/sample_sign.ppm",
        help="Path to the base image to apply effects on.",
    )
    parser.add_argument(
        "--output-path",
        default="analysis_output/figures/effects_gallery.png",
        help="Path to save the output gallery image.",
    )
    parser.add_argument(
        "--cols", type=int, default=3, help="Number of columns in the output grid."
    )
    args = parser.parse_args()

    # Create a dummy asset if it doesn't exist for easy first run
    if not os.path.exists(args.image_path):
        os.makedirs(os.path.dirname(args.image_path) or ".", exist_ok=True)
        print(f"Sample image not found, creating a dummy one at '{args.image_path}'")
        dummy_img = Image.new("RGB", (100, 100), color="red")
        dummy_img.save(args.image_path)

    visualize_effects(
        effects=args.effects,
        image_path=args.image_path,
        output_path=args.output_path,
        cols=args.cols,
    )


if __name__ == "__main__":
    main()
