import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
import yaml
import argparse

# Make sure to import your model definition
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN

# Dictionary to store the activations
activations = {}

def get_activation(name):
    """Hook function to capture the output of a layer."""
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def visualize_activations(config_path: str, model_path: str, image_path: str, output_dir: str):
    """Loads a model and an image, and visualizes the layer activations."""
    
    # --- 1. Load Configuration and Build the Model ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_params = config.get('model', {}).get('params', {})
    if not model_params:
        raise ValueError("Model parameters not found in config file.")

    # Build the model with the correct architecture
    model = SimpleCNN(**model_params)
    
    # Load the trained weights into the correctly-structured model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Model loaded from {model_path} using architecture from {config_path}")
    
    # --- 2. Register Hooks ---
    # This part remains the same. It correctly targets the hidden layers.
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.pool.register_forward_hook(get_activation('pool1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    
    # --- 3. Prepare the Input Image ---
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0) # Add a batch dimension
    
    # --- 4. Perform Forward Pass ---
    with torch.no_grad():
        output = model(image_tensor)
    
    print("Forward pass complete. Activations captured.")
    
    # --- 5. Plot the Activations ---
    for name, feature_map in activations.items():
        # The feature map is a 4D tensor (batch, channels, height, width)
        # We take the first item in the batch
        feature_map = feature_map[0]
        
        # Make a grid of the channels
        grid = torchvision.utils.make_grid(
            feature_map.unsqueeze(1), # Add a dimension for grayscale
            nrow=4,                  # Adjust number of columns in the grid
            normalize=True,
            pad_value=1
        )
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"Activations for Layer: '{name}'\nShape: {list(feature_map.shape)}")
        plt.axis('off')
        
        save_path = f"{output_dir}/{name}_activations.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Activation plot for '{name}' saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize layer activations for a given image.")
    # ADD the config path argument
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment config file (.yaml).")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model (.pth).")
    parser.add_argument('--image-path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output-dir', type=str, default='activations_output', help="Directory to save output images.")
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    visualize_activations(args.config, args.model_path, args.image_path, args.output_dir)

