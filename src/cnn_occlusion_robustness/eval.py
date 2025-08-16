import torch
import argparse
import json
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from cnn_occlusion_robustness.data.gtsrb import GTSRBDataset
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN
from cnn_occlusion_robustness.train import get_effect, AugmentedDataset 

def evaluate_model(model_path: str, data_dir: str, test_effect: str, output_path: str):
    """Evaluates a single model on a single test condition."""
    
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. Load Model ---
    model = SimpleCNN(num_classes=43).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # --- 3. Prepare Dataset ---
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    # NOTE: You should create a dedicated test set, not reuse the full dataset.
    # For now, we'll simulate it.
    full_dataset = GTSRBDataset(root_dir=data_dir, transform=transform)
    effect = get_effect(test_effect)
    augmented_test_dataset = AugmentedDataset(full_dataset, effect)
    test_loader = DataLoader(augmented_test_dataset, batch_size=128, shuffle=False)
    
    # --- 4. Run Evaluation ---
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating on '{test_effect}'"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    print(f"Accuracy on '{test_effect}': {accuracy:.4f}")
    
    # --- 5. Save Result ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        'model_path': model_path,
        'test_effect': test_effect,
        'accuracy': accuracy,
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model (.pth).")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the GTSRB test dataset.")
    parser.add_argument('--test-effect', type=str, required=True, help="Occlusion effect to apply during testing.")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the evaluation result JSON.")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_dir, args.test_effect, args.output_path)
