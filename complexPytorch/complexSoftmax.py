
"""
Implementations of softmax functions for complex numbers.

The class implementations are just boilerplate-wrappers to use them as nn.Modules.
"""

import torch
import torch.nn as nn


# softmax implementations inspired by https://complex-valued-neural-networks.readthedocs.io/en/latest/activations/real_output.html
# Numbers declare the decreasing test-accuracy order in the evaluation on cifar10
def softmax_real_with_abs(x, dim=-1): #4
    return torch.softmax(x.abs(), dim=dim)

def softmax_real_with_avg(x, dim=-1): #1
    return (torch.softmax(x.real, dim=dim) + torch.softmax(x.imag, dim=dim)) * 0.5

def softmax_real_with_mult(x, dim=-1): #5
    return torch.softmax(x.real, dim=dim) * torch.softmax(x.imag, dim=dim)

def softmax_of_softmax_real_with_mult(x, dim=-1): #6
    return torch.softmax(torch.softmax(x.real, dim=dim) * torch.softmax(x.imag, dim=dim), dim=dim)

def softmax_of_softmax_real_with_avg(x, dim=-1): #2
    return torch.softmax(torch.softmax(x.real, dim=dim) + torch.softmax(x.imag, dim=dim), dim=dim)

def softmax_real_with_avg_polar(x, dim=-1): #3
    return (torch.softmax(x.abs(), dim=dim) + torch.softmax(x.angle(), dim=dim)) * 0.5

def softmax_real_with_polar(x, dim=-1): #7
    return torch.softmax(x.angle(), dim=dim)

def softmax_complex(x, dim=-1):
    return torch.exp(x)/torch.exp(x).sum(dim=dim, keepdim=True)

# for numerical stability, subtract maximum real and imaginary part before exp
def softmax_complex_max_diff(x, dim=-1):
    z = x - x.real.max(dim=dim, keepdim=True).values - 1j * x.imag.max(dim=dim, keepdim=True).values
    return torch.exp(z)/torch.exp(z).sum(dim=dim, keepdim=True)

def softmax_complex_split(x, dim=-1):
    return torch.softmax(x.real, dim=dim) + 1j * torch.softmax(x.imag, dim=dim)


class SoftmaxRealWithAbs(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxRealWithAbs, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_real_with_abs(x, dim=self.dim)

class SoftmaxRealWithAvg(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxRealWithAvg, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_real_with_avg(x, dim=self.dim)

class SoftmaxRealWithMult(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxRealWithMult, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_real_with_mult(x, dim=self.dim)

class SoftmaxOfSoftmaxRealWithMult(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxOfSoftmaxRealWithMult, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_of_softmax_real_with_mult(x, dim=self.dim)

class SoftmaxOfSoftmaxRealWithAvg(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxOfSoftmaxRealWithAvg, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_of_softmax_real_with_avg(x, dim=self.dim)

class SoftmaxRealWithAvgPolar(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxRealWithAvgPolar, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_real_with_avg_polar(x, dim=self.dim)

class SoftmaxRealWithPolar(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxRealWithPolar, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_real_with_polar(x, dim=self.dim)

class SoftmaxComplex(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxComplex, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_complex(x, dim=self.dim)

class SoftmaxComplexMaxDiff(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxComplexMaxDiff, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_complex_max_diff(x, dim=self.dim)

class SoftmaxComplexSplit(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxComplexSplit, self).__init__()
        self.dim = dim
    def forward(self, x):
        return softmax_complex_split(x, dim=self.dim)

