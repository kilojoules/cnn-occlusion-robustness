import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from typing import List, Tuple

class GTSRBDataset(Dataset):
    """
    GTSRB Dataset class that works with the folder structure,
    where each subfolder in the 'Images' directory is a class.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.loader = default_loader
        self.samples = self._make_dataset()

        if not self.samples:
            raise RuntimeError(
                f"No image files found in {root_dir}. "
                f"Please check that the directory structure is correct (e.g., root_dir/00000/img.ppm)."
            )

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Scans the directory for image paths and labels."""
        samples = []
        # Get the class folder names (e.g., '00000', '00001', ...)
        class_folders = sorted([d.name for d in os.scandir(self.root_dir) if d.is_dir()])
        
        for class_index, class_folder_name in enumerate(class_folders):
            class_path = os.path.join(self.root_dir, class_folder_name)
            # Find all .ppm files in the class folder
            for img_name in os.listdir(class_path):
                if img_name.endswith('.ppm'):
                    img_path = os.path.join(class_path, img_name)
                    item = (img_path, class_index) # Use the sorted index as the label
                    samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = self.loader(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
