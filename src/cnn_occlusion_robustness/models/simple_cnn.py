# src/models/simple_cnn.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN for GTSRB classification, inspired by LeNet.
    It expects 32x32x3 input images.
    """
    def __init__(self, num_classes=43):
        super(SimpleCNN, self).__init__()
        # First convolutional layer (sees 32x32x3 image)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # The input features to fc1 depend on the output of conv2 and pooling
        # Input: 32x32 -> Conv1(5x5) -> 28x28 -> Pool(2x2) -> 14x14
        # -> Conv2(5x5) -> 10x10 -> Pool(2x2) -> 5x5.
        # So, the flattened size is 16 channels * 5 * 5 = 400
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation here, as CrossEntropyLoss will apply softmax
        return x
