import os
import pytest
from cnn_occlusion_robustness.data.gtsrb import GTSRBDataset
from torchvision import transforms

@pytest.fixture
def dummy_data_dir(tmp_path):
    # Create a minimal fake dataset structure
    class_dir = tmp_path / "00000"
    class_dir.mkdir(parents=True)
    # Make a tiny dummy image
    from PIL import Image
    img = Image.new("RGB", (32, 32), color="red")
    img.save(class_dir / "00000.ppm")
    return tmp_path

def test_dataset_loading(dummy_data_dir):
    transform = transforms.ToTensor()
    ds = GTSRBDataset(root_dir=str(dummy_data_dir), transform=transform)
    assert len(ds) == 1
    img, label = ds[0]
    assert img.shape[0] == 3   # RGB
    assert label == 0          # first class

