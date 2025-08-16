import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import argparse
import json
import os

# Import modules from within the same package
from cnn_occlusion_robustness.data.gtsrb import GTSRBDataset
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN

# Import the external dependency
from camera_occlusion import Rain, Dust, Effect


# --- The AugmentedDataset class is defined at the top level to be "pickleable" ---
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, effect):
        self.dataset = dataset
        self.effect = effect

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_tensor, label = self.dataset[idx]
        # Fast path for no-op
        if isinstance(self.effect, NoOp):
            return image_tensor, label
        # Convert tensor (C,H,W)->(H,W,C) uint8 for augmentation
        image_np = (image_tensor.permute(1, 2, 0).contiguous().numpy() * 255).astype(
            "uint8"
        )
        augmented_np = self.effect(image_np)
        augmented_tensor = torch.from_numpy(augmented_np).permute(2, 0, 1) / 255.0
        return augmented_tensor.float(), label


def apply_mixed_light(img):
    """Applies a light mix of rain and dust effects."""
    effect = Dust(num_specks=1200, splotch_opacity=0.08)(
        Rain(num_drops=20, radius_range=(2, 4))(img)
    )
    return effect


def apply_mixed_heavy(img):
    """Applies a heavy mix of rain and dust effects."""
    effect = Dust(num_specks=4200, num_scratches=4, splotch_opacity=0.14)(
        Rain(num_drops=120, radius_range=(4, 8))(img)
    )
    return effect


class NoOp:
    """Numpy in -> numpy out no-op."""

    def __call__(self, img_np):
        return img_np


def get_effect(effect_name: str) -> Effect:
    """Factory function to get an effect instance from its name."""
    effects = {
        "none": NoOp(),
        "light_rain": Rain(num_drops=15, radius_range=(2, 4), magnification=1.03),
        "moderate_rain": Rain(num_drops=50, radius_range=(3, 6), magnification=1.08),
        "heavy_rain": Rain(num_drops=100, radius_range=(4, 8), magnification=1.15),
        "light_dust": Dust(
            num_specks=20, speck_opacity=0.4, num_scratches=1, splotch_opacity=0.1
        ),
        "moderate_dust": Dust(
            num_specks=40, speck_opacity=0.4, num_scratches=3, splotch_opacity=0.05
        ),
        "heavy_dust": Dust(
            num_specks=50, speck_opacity=0.8, num_scratches=5, splotch_opacity=0.02
        ),
        "mixed_light": apply_mixed_light,
        "mixed_heavy": apply_mixed_heavy,
    }
    if effect_name not in effects:
        raise ValueError(
            f"Unknown effect: {effect_name}. Available: {list(effects.keys())}"
        )
    return effects[effect_name]


def main():
    """
    Main function to handle training, with parameters loaded from a YAML config file.
    """
    parser = argparse.ArgumentParser(
        description="Train a CNN on GTSRB with occlusion effects."
    )

    # The script now takes the config file and the specific details for this run.
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config file (.yaml).",
    )
    parser.add_argument(
        "--train-effect",
        type=str,
        required=True,
        help="The specific augmentation effect for this training run.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Full path where the trained model (.pth) will be saved.",
    )

    args = parser.parse_args()

    # --- 1. Load Configuration from YAML ---
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        return

    print("--- Loaded Configuration ---")
    print(json.dumps(config, indent=2))

    # Extract parameters from the loaded config
    data_dir = config["data_dir"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # --- 2. Setup Device, Data, and Model ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Get the specific augmentation effect for this run
    train_effect_instance = get_effect(args.train_effect)

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    full_dataset = GTSRBDataset(root_dir=data_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    augmented_train_dataset = AugmentedDataset(train_dataset, train_effect_instance)

    train_loader = DataLoader(
        augmented_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Get the model architecture details from the config
    model_config = config.get("model")
    if not model_config:
        raise ValueError("Model configuration not found in YAML file.")

    # Instantiate the model using the parameters from the config
    # The ** operator unpacks the dictionary into keyword arguments
    model = SimpleCNN(**model_config["params"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 3. Training and Validation Loop ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"--- Starting training for '{args.train_effect}' for {epochs} epochs ---")

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
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
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.4f}"}
            )

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)

        # Validation Phase
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.4f}"}
                )

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), args.save_path)
            print(
                f"New best model saved to {args.save_path} with accuracy: {best_val_acc:.4f}"
            )

    # --- 4. Save Final Results ---
    final_results = {
        "config_file": args.config,
        "training_effect": args.train_effect,
        "best_val_acc": best_val_acc,
        "history": history,
    }
    history_path = os.path.splitext(args.save_path)[0] + "_results.json"
    with open(history_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Training complete. Final results saved to {history_path}")


if __name__ == "__main__":
    main()
