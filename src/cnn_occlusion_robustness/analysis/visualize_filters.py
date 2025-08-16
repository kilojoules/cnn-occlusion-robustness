import torch
import matplotlib.pyplot as plt
import torchvision
import argparse
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN

def visualize_conv1_filters(model_path: str, output_path: str):
    """Loads a trained model and visualizes the weights of its first convolutional layer."""
    
    # --- 1. Load the Model ---
    # Instantiate the model architecture
    model = SimpleCNN(num_classes=43)
    
    # Load the trained weights
    # Use map_location to load onto CPU if you don't have a GPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Model loaded from {model_path}")
    
    # Set the model to evaluation mode
    model.eval()
    
    # --- 2. Access the Weights ---
    # Get the weights of the first convolutional layer
    weights = model.conv1.weight.data
    
    # The weights are a 4D tensor: (out_channels, in_channels, height, width)
    # We need to normalize them to the [0, 1] range for visualization
    weights_min, weights_max = torch.min(weights), torch.max(weights)
    weights = (weights - weights_min) / (weights_max - weights_min)
    
    print(f"Filter weights shape: {weights.shape}") # Should be torch.Size([6, 3, 5, 5])
    
    # --- 3. Plot the Filters ---
    # Use torchvision's make_grid to arrange the filters in a grid
    grid = torchvision.utils.make_grid(weights, nrow=3, padding=1)
    
    plt.figure(figsize=(6, 4))
    plt.imshow(grid.permute(1, 2, 0)) # Permute from (C, H, W) to (H, W, C) for matplotlib
    plt.title("Conv1 Learned Filters (Input-Agnostic)")
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Filter visualization saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the filters of a trained SimpleCNN model.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model (.pth).")
    parser.add_argument('--output-path', type=str, default='conv1_filters.png', help="Path to save the output image.")
    args = parser.parse_args()
    
    visualize_conv1_filters(args.model_path, args.output_path)
