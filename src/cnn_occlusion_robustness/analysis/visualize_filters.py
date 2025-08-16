import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN
import yaml

def visualize_filters(config_path: str, model_path: str, layer_name: str, output_path: str):
    """Loads a trained model and visualizes the weights of a specified convolutional layer."""
    
    # --- 1. Load Configuration and Build the Model ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_params = config.get('model', {}).get('params', {})
    if not model_params:
        raise ValueError("Model parameters not found in config file.")

    # Build the model with the correct architecture
    model = SimpleCNN(**model_params)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Model loaded from {model_path} using architecture from {config_path}")
    
    # --- 2. Access the Weights of the Specified Layer ---
    try:
        # Dynamically get the layer from the model by its name
        layer = getattr(model, layer_name)
        if not isinstance(layer, torch.nn.Conv2d):
            raise TypeError("Selected layer is not a Conv2d layer.")
    except AttributeError:
        print(f"Error: Layer '{layer_name}' not found in the model.")
        return
        
    weights = layer.weight.data
    
    # Normalize for visualization. For conv2, we need to handle single-channel inputs.
    if weights.size(1) > 3: # If input channels are not RGB (e.g., for conv2)
        # We can't visualize 6 input channels directly, so we visualize each filter's response
        # to its first input channel as a grayscale image.
        weights = weights[:, 0, :, :].unsqueeze(1)

    weights_min, weights_max = torch.min(weights), torch.max(weights)
    weights = (weights - weights_min) / (weights_max - weights_min)
    
    print(f"Filter weights shape for '{layer_name}': {layer.weight.data.shape}")
    
    # --- 3. Plot the Filters ---
    grid = torchvision.utils.make_grid(weights, nrow=4, padding=1)
    
    plt.figure(figsize=(8, 8))
    # Permute to (H, W, C) for matplotlib. Handle grayscale case.
    if grid.shape[0] == 1:
        plt.imshow(grid.squeeze(), cmap='gray')
    else:
        plt.imshow(grid.permute(1, 2, 0))
        
    plt.title(f"Learned Filters for Layer: '{layer_name}'")
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Filter visualization saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the filters of a trained SimpleCNN model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment config file (.yaml).")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model (.pth).")
    parser.add_argument('--layer-name', type=str, default='conv1', help="Name of the conv layer to visualize (e.g., 'conv1', 'conv2').")
    parser.add_argument('--output-path', type=str, default='filters.png', help="Path to save the output image.")
    args = parser.parse_args()
    
    visualize_filters(args.config, args.model_path, args.layer_name, args.output_path)
