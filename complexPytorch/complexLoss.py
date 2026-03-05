
"""
Complex Losses, some losses already work with complex inputs in pytorch.
"""

import torch
import torch.nn as nn

class CrossEntropyComplex(nn.Module):
    """Autograd works only with real scalars, so a complex loss doesnt work directly."""
    def __init__(self, softmax=nn.Identity(), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.softmax = softmax

    def forward(self, x, y):
        onehot = torch.nn.functional.one_hot(y.long(), num_classes=x.shape[1])
        return - torch.sum((onehot + 1j * onehot) * torch.log(self.softmax(x))) / x.shape[0]

class CrossEntropyComplexTwice(nn.Module):
    def forward(self, x, y):
        return 0.5 * (nn.functional.cross_entropy(x.real, y.long()) + nn.functional.cross_entropy(x.imag, y.long()))


class NegativeLogLossComplex(nn.Module):
    def forward(self, mean, variance, y):
        var_real, var_imag = variance

        err_real = mean.real - y.real
        err_imag = mean.imag - y.imag

        nll_real = 0.5 * (err_real ** 2 / var_real + torch.log(var_real) + 1e-6)
        nll_imag = 0.5 * (err_imag ** 2 / var_imag + torch.log(var_imag) + 1e-6)

        return (nll_real + nll_imag).mean()
