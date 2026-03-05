import torch
from torch import nn
import torch.nn.functional as F


class RegressionNN(nn.Module):
    def __init__(self, in_features):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.relu = nn.ReLU()

        self.mean_head = nn.Linear(16, 2)
        self.var_head = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.view_as_real(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        mean = self.mean_head(x)
        var = self.var_head(x)


        return mean, var


class LinearRegressionNN(nn.Module):
    def __init__(self, lag, in_features=2):
        super(LinearRegressionNN, self).__init__()
        self.fc1 = nn.Linear(lag*in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.mean = nn.Linear(16, 2)
        self.variance = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.view_as_real(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        mean = self.mean(x)
        mean = torch.complex(mean[:, 0], mean[:, 1])
        variance = self.variance(x)
        var_real = F.softplus(variance[:, 0])
        var_imag = F.softplus(variance[:, 1])
        return mean, (var_real, var_imag)
