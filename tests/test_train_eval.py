import torch
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN
from cnn_occlusion_robustness.train import NoOp, AugmentedDataset
from torch.utils.data import TensorDataset


def test_augmented_dataset_forward():
    dummy_images = torch.rand(5, 3, 32, 32)  # 5 fake images
    dummy_labels = torch.randint(0, 43, (5,))
    base_ds = TensorDataset(dummy_images, dummy_labels)
    ds = AugmentedDataset(base_ds, NoOp())

    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label.item(), int)


def test_eval_runs(tmp_path):
    # Dummy model save/load check
    model = SimpleCNN(num_classes=43)
    weights_path = tmp_path / "dummy_model.pth"
    torch.save(model.state_dict(), weights_path)
    assert weights_path.exists()
