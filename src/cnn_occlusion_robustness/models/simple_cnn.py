import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(
        self, num_classes=43, channels=[3, 6, 16], kernel_size=5, fc_features=[120, 84]
    ):
        super(SimpleCNN, self).__init__()

        # Use parameters from the config
        self.conv1 = nn.Conv2d(
            in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size
        )
        self.conv2 = nn.Conv2d(
            in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Dynamically determine the input size for fc1 ---
        # This is the robust way to handle changing conv layers
        dummy_input = torch.randn(1, channels[0], 32, 32)  # A sample input tensor
        conv_output_size = self._get_conv_output_size(dummy_input)

        self.fc1 = nn.Linear(in_features=conv_output_size, out_features=fc_features[0])
        self.fc2 = nn.Linear(in_features=fc_features[0], out_features=fc_features[1])
        self.fc3 = nn.Linear(in_features=fc_features[1], out_features=num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def _get_conv_output_size(self, x):
        """Helper function to calculate the flattened size after conv layers."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.flatten().shape[0]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Use view with -1 to be flexible

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
