import torch
from torch import nn
import torch.nn.functional as F


class RegressionNN(nn.Module):
    def __init__(self, in_features):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.relu1 = nn.ReLU()

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
    def __init__(self, lag, hidden_size=64, num_layers=1):
        super(LinearRegressionNN, self).__init__()

        # LSTM preserves temporal ordering of the lag window
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.mean_head = nn.Linear(16, 2)
        self.variance_head = nn.Linear(16, 3)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        # x: (batch, lag) complex → (batch, lag, 2) real
        x = torch.view_as_real(x)              # (batch, lag, 2)

        # LSTM over lag dimension
        out, _ = self.lstm(x)                  # (batch, lag, hidden_size)
        out = out[:, -1, :]                    # take last timestep (batch, hidden_size)

        out = self.act1(self.fc1(out))
        out = self.act2(self.fc2(out))

        mean = self.mean_head(out)
        mean = torch.complex(mean[:, 0], mean[:, 1])

        variance = self.variance_head(out)
        var_real = F.softplus(variance[:, 0])
        var_imag = F.softplus(variance[:, 1])
        rho = variance[:, 2]
        rho = torch.tanh(rho)

        return mean, (var_real, var_imag), rho
