import torch
from torch import nn


class TemperatureScaling(nn.Module):
    """
    Temperature scaling module for model calibration.
    Learns a single temperature parameter to scale logits.
    """

    def __init__(self, device='cuda'):
        super(TemperatureScaling, self).__init__()
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.to(self.device)


    def forward(self, logits):
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def fit(self, model,val_loader, lr=0.01, max_iter=50):
        """
        Tune the temperature parameter using validation set.

        Args:
            model
            val_loader
            lr: Learning rate for optimization
            max_iter: Maximum number of optimization iterations
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        labels = []
        logits = []


        for x,y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            out = model.forward(x)
            out = torch.tensor(out).to(self.device)
            logits.append(out)
            labels.append(y)

        logits = torch.cat(logits, dim=0).detach().to(self.device)
        labels = torch.cat(labels, dim=0).to(self.device)



        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        return self.temperature.item()
