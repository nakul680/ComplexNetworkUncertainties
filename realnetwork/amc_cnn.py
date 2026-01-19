from torch import nn
import torch.nn.functional as F

from complexnetwork.CDSCNN import *


class AMC_CNN(nn.Module):
    """
    CNN-based Automatic Modulation Classification Network
    Input: [batch_size, 2, 128] (2 channels: I/Q, 128 time steps)
    Output: [batch_size, 11] (11 modulation classes)
    """
    is_complex = False

    def __init__(self, num_classes=11):
        super(AMC_CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)  # 128 -> 64

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)  # 64 -> 32

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)  # 32 -> 16

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 16, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        # self.dropout2 = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [batch, 2, 128]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x

    def compute_features(self, x):
        """
        Extract feature embeddings before final classification layer
        Args:
                complex_params = sum(p.numel() for p in complex_model.parameters())x: Input tensor [batch, 2, 128]
        Returns:
            Feature embeddings [batch, 128]
        """
        # Set to eval mode to disable dropout
        # was_training = self.training
        # self.eval()
        #
        # with torch.no_grad():
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
#        x = self.dropout2(F.relu(x))

        # # Restore original training state
        # if was_training:
        #     self.train()
        #
        return x


class AMC_MLP(nn.Module):
    """
    MLP-based Automatic Modulation Classification Network
    Input: [batch_size, 2, 128] (2 channels: I/Q, 128 time steps)
    Output: [batch_size, 11] (11 modulation classes)
    """

    def __init__(self, num_classes=11, hidden_sizes=[512, 256, 128], dropout=0.5):
        super(AMC_MLP, self).__init__()

        input_size = 2 * 128  # flatten 2 channels x 128 length
        self.flatten = nn.Flatten()

        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, num_classes))  # output layer

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, 2, 128]
        x = self.flatten(x)  # [batch, 256]
        x = self.mlp(x)
        # x = torch.softmax(x, dim=1)
        return x
