import torch
from torch import nn
import torch.nn.functional as F

from complexPytorch.complexLayers import Cardioid, ComplexLSTM, ComplexRNN


class ComplexRegressionLSTM(nn.Module):
    def __init__(self, lag, hidden_size=32, num_layers=1):
        super(ComplexRegressionLSTM, self).__init__()

        self.lstm = ComplexLSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 16, dtype=torch.complex64)
        self.fc2 = nn.Linear(16, 8, dtype=torch.complex64)
        self.relu1 = Cardioid()
        self.relu2 = Cardioid()

        self.mean_head = nn.Linear(8, 1, dtype=torch.complex64)
        self.var_head = nn.Linear(16, 3)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))

        mean = self.mean_head(x).squeeze(-1)
        x_real = torch.view_as_real(x).flatten(1)
        var = self.var_head(x_real)
        var_real = F.softplus(var[:, 0])
        var_imag = F.softplus(var[:, 1])
        rho = var[:, 2]
        rho = torch.tanh(rho)

        return mean, (var_real, var_imag), rho


class ComplexRegressionNN(nn.Module):
    def __init__(self, lag, hidden_size=16, num_layers=1):
        super().__init__()

        # Complex RNN using GRU cells
        self.rnn = ComplexRNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Complex feedforward layers
        self.fc1 = nn.Linear(hidden_size, 8, dtype=torch.complex64)
        self.fc2 = nn.Linear(8, 4, dtype=torch.complex64)
        self.relu1 = Cardioid()
        self.relu2 = Cardioid()

        # Mean head (complex output)
        self.mean_head = nn.Linear(4, 1, dtype=torch.complex64)

        # Variance/correlation head (real output)
        self.var_head = nn.Linear(8, 3)  # [log_var_real, log_var_imag, rho_logit]

    def forward(self, x):
        # x shape: (batch, lag) -> (batch, lag, 1)
        x = x.unsqueeze(-1)

        # RNN processing
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Last timestep

        # Complex feedforward
        x = self.relu1(self.fc1(x))
        x_for_var = x  # Save for variance head
        x = self.relu2(self.fc2(x))

        # Mean prediction (complex)
        mean = self.mean_head(x).squeeze(-1)

        # Variance and correlation (real)
        x_real = torch.view_as_real(x).flatten(1)  # (batch, 32)
        var_params = self.var_head(x_real)

        log_var_real = var_params[:, 0]
        log_var_imag = var_params[:, 1]
        rho_logit = var_params[:, 2]

        var_real = torch.exp(log_var_real) + 1e-6
        var_imag = torch.exp(log_var_imag) + 1e-6
        rho = torch.tanh(rho_logit)

        return mean, (var_real, var_imag), rho
