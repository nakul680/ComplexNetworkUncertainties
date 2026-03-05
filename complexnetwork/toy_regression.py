import torch
from torch import nn

from complexPytorch import ComplexLinear, ComplexReLU


class ComplexRegressionNN(nn.Module):
    def __init__(self, in_features):
        super(ComplexRegressionNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 64, dtype=torch.complex64)
        self.fc2 = nn.Linear(64, 32, dtype=torch.complex64)
        self.fc3 = nn.Linear(32, 16, dtype=torch.complex64)
        self.relu = ComplexReLU()

        self.mean_head = nn.Linear(16, 1, dtype=torch.complex64)
        self.var_head = nn.Linear(16, 1, dtype=torch.complex64)


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # x_real = torch.view_as_real(x)
        # x_real = x_real.flatten(start_dim=1)
        mean = self.mean_head(x)
        var = self.var_head(x)

        return mean, var