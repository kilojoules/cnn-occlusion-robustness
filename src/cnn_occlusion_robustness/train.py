# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import argparse
import json
import os

# Import your custom modules
from data.gtsrb import GTSRBDataset
from models.simple_cnn import SimpleCNN # Assumes this file exists
from camera_occlusion import Rain, Dust, Effect # Your installed library!

def get_effect(effect_name: str) -> Effect:
    """Factory function to get an effect instance from its name."""
    effects = {
        'none': nn.Identity(),
        'light_rain': Rain(num_drops=15, radius_range=(2, 4)),
        'heavy_rain': Rain(num_drops=100, radius_range=(4, 8)),
        'light_dust': Dust(num_specks=1000, splotch_opacity=0.08),
        'heavy_dust': Dust(num_specks=5000, num_scratches=5, splotch_opacity=0.15),
    }
    if effect_name not in effects:
        raise ValueError(f"Unknown effect: {effect_name}. Available: {list(effects.keys())}")
    return effects[effect_name]

def main(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define Augmentation
    train_effect = get_effect(args.train_effect)
    
    # 3. Setup DataLoaders
    # Note: Effects are applied after converting to tensor for numpy compatibility
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    full_dataset = GTSRBDataset(root_dir=args.data_dir, train=True, transform=transform)
    
    # Split dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply the custom augmentation only to the training set via a wrapper
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, effect):
            self.dataset = dataset
            self.effect = effect

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image_tensor, label = self.dataset[idx]
            # Convert to numpy HWC for cv2-based effects
            image_np = image_tensor.permute(1, 2, 0).numpy() * 255
            image_np = image_np.astype('uint8')
            
            # Apply effect
            augmented_np = self.effect(image_np)
            
            # Convert back to tensor CHW
            augmented_tensor = torch.from_numpy(augmented_np).permute(2, 0, 1) / 255.0
            return augmented_tensor.float(), label

    augmented_train_dataset = AugmentedDataset(train_dataset, train_effect)
    
    train_loader = DataLoader(augmented_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 4. Initialize Model, Loss, Optimizer
    model = SimpleCNN(num_classes=43).to(device) # You need to create this model file
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 5. Training Loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"Starting training for '{args.train_effect}' for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(correct/total):.4f}"})
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{(correct/total):.4f}"})

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved to {args.save_path} with accuracy: {best_val_acc:.4f}")

    # 6. Save Final Results
    final_results = {
        'config': vars(args),
        'best_val_acc': best_val_acc,
        'history': history
    }
    
    history_path = os.path.splitext(args.save_path)[0] + '_results.json'
    with open(history_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Training complete. Final results saved to {history_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN on GTSRB with occlusion effects.")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to the GTSRB dataset root directory.")
    parser.add_argument('--train-effect', type=str, default='none', help="Augmentation effect to apply during training.")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--learning-rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--save-path', type=str, required=True, help="Path to save the trained model (.pth).")
    
    args = parser.parse_args()
    
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    main(args)
