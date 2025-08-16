import torch
from cnn_occlusion_robustness.models.simple_cnn import SimpleCNN


def test_forward_pass():
    model = SimpleCNN(num_classes=43)
    x = torch.randn(4, 3, 32, 32)  # batch of 4 RGB images
    y = model(x)
    assert y.shape == (4, 43)
