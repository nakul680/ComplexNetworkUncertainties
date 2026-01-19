from typing import Optional

import torch, warnings
from torch.nn import (
    Module, Parameter, init,
    Conv2d, ConvTranspose2d, Linear, LSTM, GRU,
    BatchNorm1d, BatchNorm2d,
    PReLU
)
import torch.nn as nn

from complexPytorch.complexFunctions import (
    complex_relu, complex_gelu,
    complex_tanh,
    complex_sigmoid,
    complex_max_pool2d, complex_max_pool1d,
    complex_avg_pool2d,
    complex_adaptive_avg_pool2d,
    complex_dropout,
    complex_dropout2d,
    complex_opposite
)

from complexPytorch.complexSoftmax import (
    softmax_complex_split
)


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) \
        + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


# Activation Functions, Pooling, Dropout
class ComplexCardioid(Module):
    @staticmethod
    def forward(inp):
        mag = torch.abs(inp)
        cos_theta = inp.real / (mag + 1e-4)
        gate = 0.5 * (1.0 + cos_theta)
        return gate * inp


class ComplexReLU(Module):  # best of the tested relus (CReLU, zReLU, modReLU)
    @staticmethod
    def forward(inp):  # relu(x.real) + relu(x.imag)
        return complex_relu(inp)


class zReLU(Module):
    """zReLU from https://openreview.net/forum?id=H1T2hmZAb"""

    @staticmethod
    def forward(inp):
        # some implementation version, most have problems with backpropagation because of 0-grads:
        # last one worked

        # return torch.relu(inp.real) * ((1+torch.sign(inp.imag)) / 2) + 1j * torch.relu(inp.imag) * ((1+torch.sign(inp.real)) / 2)

        # return torch.relu(inp.real) * (2*(torch.min(torch.sign(inp.imag), torch.tensor(-0.5, requires_grad=True)) + 1 )) + \
        #       1j * torch.relu(inp.imag) * (2* (torch.min(torch.sign(inp.real), torch.tensor(-0.5, requires_grad=True)) + 1))

        # return torch.where((inp.real >= 0) & (inp.imag >= 0), inp, 0 + 0j)

        # return torch.where( #same as (relu(x.real) > 0) & (relu(x.imag) > 0)
        #             (inp.angle() >= 0) & 
        #             (inp.angle() <= torch.pi/2),
        #             inp, 0 + 0j)

        if inp.is_complex():
            return inp * ((0 < inp.angle()) * (inp.angle() < torch.pi / 2)).float()
        else:
            return torch.relu(inp)


class modReLU(nn.Module):
    """modReLU from https://arxiv.org/abs/1511.06464, added by Luca Hinkamp"""

    def __init__(self, data_shape=(1,), init_b=0., *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # init values between -1 and 1 should be good, otherwise its hard for the optimizer to get to the right value range
        self.b = nn.Parameter(torch.empty(data_shape).fill_(init_b), requires_grad=True)

    def forward(self, x):
        return nn.functional.relu(x.abs() + self.b) * torch.exp(1j * x.angle())


class ComplexPReLU(Module):
    def __init__(self):
        super().__init__()
        self.r_prelu = PReLU()
        self.i_prelu = PReLU()

    def forward(self, inp):
        return self.r_prelu(inp.real) + 1j * self.i_prelu(inp.imag)


class ComplexSigmoid(Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid(inp)


class ComplexTanh(Module):
    @staticmethod
    def forward(inp):
        return complex_tanh(inp)


class ComplexGELU(Module):
    @staticmethod
    def forward(inp):
        return complex_gelu(inp)


class ComplexDropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.p)
        else:
            return input


class ComplexDropout2d(Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complex_dropout2d(inp, self.p)
        else:
            return inp


class ComplexMaxPool2d(Module):
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
    ):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool2d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class ComplexMaxPool1d(Module):
    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
    ):
        super(ComplexMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool1d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class ComplexAvgPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(ComplexAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, inp):
        return complex_avg_pool2d(inp, kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  ceil_mode=self.ceil_mode, count_include_pad=self.count_include_pad,
                                  divisor_override=self.divisor_override)


class ComplexAdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, inp):
        return complex_adaptive_avg_pool2d(inp, self.output_size)


# Main Layers, Linear, Conv2d, Norm-Layers

# other base-layers work same to conv2d / linear in base-pytorch, so just use them with complex-dtype in init

class ComplexConv2d(Conv2d):
    """
    Normal way to make a complex layer. \n
    Autograd works with complex, so no 2 seperate layers needed.\n
    Needs to define dtype as complex bevor layer-init to get a normally initialized imaginary part.
    """

    def __init__(self, *args, **kwargs) -> None:
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.complex64
        super().__init__(*args, **kwargs)


class ComplexConv2dTwice(Module):
    """
    Naive approach using 2 real layers.
    Not mathematically correct.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv_i = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, inp):
        return apply_complex(self.conv_r, self.conv_i, inp)


class ComplexConvTranspose2d(ConvTranspose2d):
    """
    Normal way to make a complex layer. \n
    Autograd works with complex, so no 2 seperate layers needed.\n
    Needs to define dtype as complex bevor layer-init to get a normally initialized imaginary part.
    """

    def __init__(self, *args, **kwargs) -> None:
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.complex64
        super().__init__(*args, **kwargs)


class ComplexConvTranspose2dTwice(Module):
    """
    Naive approach using 2 real layers.
    Not mathematically correct.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
            dilation=1,
            padding_mode="zeros",
    ):
        super().__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                           output_padding, groups, bias, dilation, padding_mode)

    def forward(self, inp):
        return apply_complex(self.conv_tran_r, self.conv_tran_i, inp)


class ComplexLinear(Linear):
    """
    Normal way to make a complex linear layer. \n
    Autograd works with complex, so no 2 seperate layers needed.\n
    Needs to define dtype as complex bevor layer-init to get a normally initialized imaginary part.
    """

    def __init__(self, *args, **kwargs) -> None:
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.complex64
        super().__init__(*args, **kwargs)


class ComplexLinearTwice(Module):
    """
    Naive approach using 2 real layers.
    Not mathematically correct.
    """

    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__()
        self.fc_r = Linear(in_features, out_features, *args, **kwargs)
        self.fc_i = Linear(in_features, out_features, *args, **kwargs)

    def forward(self, inp):
        return apply_complex(self.fc_r, self.fc_i, inp)


class CombineComplexLinear(nn.Module):
    """
    Concat real and imaginary part in one double-sized real layer to get real output.
    """

    def __init__(self, shape_getter, outs=None, *args, **kwargs) -> None:
        """
        Provide to get the shape of the linear layer:
        - complex layer to derive shape from
        - tuple (ins, outs)
        - ins and outs as int
        """
        super().__init__(*args, **kwargs)

        if isinstance(shape_getter, nn.Module):
            outs, ins = shape_getter.weight.shape
        elif hasattr(shape_getter, "__len__") and len(shape_getter) == 2:
            ins, outs = shape_getter
        elif isinstance(shape_getter, int) and outs is not None:
            ins = shape_getter
        else:
            raise ValueError("Either ComplexLinear or (ins, outs) must be provided")
        self.fc = nn.Linear(2 * ins, outs)

    def forward(self, x):
        x = torch.cat([x.real, x.imag], dim=-1)
        return self.fc(x)


class NaiveComplexLayerNorm(Module):
    """
    Naive approach to complex layer norm, perform layer norm independently on real and imaginary part.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(NaiveComplexLayerNorm, self).__init__()
        self.ln_r = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.ln_i = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, inp):
        return self.ln_r(inp.real).type(torch.complex64) + 1j * self.ln_i(
            inp.imag
        ).type(torch.complex64)


class NaiveComplexBatchNorm1d(Module):
    """
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    """

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, inp):
        return self.bn_r(inp.real).type(torch.complex64) + 1j * self.bn_i(
            inp.imag
        ).type(torch.complex64)


class NaiveComplexBatchNorm2d(Module):
    """
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    """

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, inp):
        return self.bn_r(inp.real).type(torch.complex64) + 1j * self.bn_i(
            inp.imag
        ).type(torch.complex64)


class _ComplexBatchNorm(Module):
    running_mean: Optional[torch.Tensor]

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.complex64)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):
    """
    Correct implementation of complex batch norm from https://arxiv.org/abs/1705.09792.
    
    In tests it performed equally to naive, but way slower. So use carefully.
    """

    def forward(self, inp):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                                                 float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = inp.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = inp.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, :, None, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * inp.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                        exponential_average_factor * Crr * n / (n - 1)  #
                        + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                        exponential_average_factor * Cii * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                        exponential_average_factor * Cri * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (
                      Rrr[None, :, None, None] * inp.real + Rri[None, :, None, None] * inp.imag
              ).type(torch.complex64) + 1j * (
                      Rii[None, :, None, None] * inp.imag + Rri[None, :, None, None] * inp.real
              ).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                          self.weight[None, :, 0, None, None] * inp.real
                          + self.weight[None, :, 2, None, None] * inp.imag
                          + self.bias[None, :, 0, None, None]
                  ).type(torch.complex64) + 1j * (
                          self.weight[None, :, 2, None, None] * inp.real
                          + self.weight[None, :, 1, None, None] * inp.imag
                          + self.bias[None, :, 1, None, None]
                  ).type(
                torch.complex64
            )
        return inp


class ComplexBatchNorm1d(_ComplexBatchNorm):
    """
    Correct implementation of complex batch norm from https://arxiv.org/abs/1705.09792.
    
    In tests it performed equally to naive, but way slower. So use carefully.
    """

    def forward(self, inp):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                                                 float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = inp.real.mean(dim=(0,2)).type(torch.complex64)
            mean_i = inp.imag.mean(dim=(0,2)).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean
                )

        inp = inp - mean[None, :, None]

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = inp.real.var(dim=(0,2), unbiased=False) + self.eps
            Cii = inp.imag.var(dim=(0,2), unbiased=False) + self.eps
            Cri = (inp.real.mul(inp.imag)).mean(dim=(0,2))
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                        exponential_average_factor * Crr * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                        exponential_average_factor * Cii * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                        exponential_average_factor * Cri * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = (Rrr[None, :, None] * inp.real + Rri[None, :, None] * inp.imag).type(
            torch.complex64
        ) + 1j * (Rii[None, :, None] * inp.imag + Rri[None, :, None] * inp.real).type(
            torch.complex64
        )

        if self.affine:
            inp = (
                          self.weight[:,0][None, :, None] * inp.real
                          + self.weight[:,2][None, :, None] * inp.imag
                          + self.bias[:,0][None, :, None]
                  ).type(torch.complex64) + 1j * (
                          self.weight[:,2][None, :, None] * inp.real
                          + self.weight[:,1][None, :, None] * inp.imag
                          + self.bias[:,1][None, :, None]
                  ).type(
                torch.complex64
            )

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return inp


# Special Layers, Attention, GRU, ...

class ComplexBasicMultiheadAttention(Module):
    """Basic variant of Multihead attention for complex valued data."""

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, batch_first=False, device=None,
                 dtype=torch.complex64) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if dtype is not None and not dtype.is_complex:
            warnings.warn("ComplexMultiheadAttention should be used with complex dtype, got {}".format(dtype))

        self.W_q = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.W_k = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.W_v = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.W_o = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.dropout_layer = ComplexDropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9 - 1e-9j)

        # splitted softmax make a bit more sense than combined normal complex softmax; maybe try realifying softmaxes in the future e.g. softmax of abs or real part
        attention_weights = softmax_complex_split(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        if not self.batch_first:
            # (L, B, E) -> (B, L, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len = query.size(0), query.size(1)

        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.W_o(attn_output)

        if not self.batch_first:
            # back to (L, B, E)
            output = output.transpose(0, 1)

        return output, attention_weights


# exists in plain pytorch for complex
class ComplexGRUCell(Module):
    """
    A GRU cell for complex-valued inputs
    """

    def __init__(self, input_length, hidden_length):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r1 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.linear_reset_w2 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r2 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        # update gate components
        self.linear_gate_w3 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_gate_r3 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.activation_gate = ComplexSigmoid()
        self.activation_candidate = ComplexTanh()

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_gate(x_1 + h_1)
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_gate(h_2 + x_2)
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        h_3 = r * self.linear_gate_r3(h)  # element-wise multiplication
        gate_update = self.activation_candidate(x_3 + h_3)
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1 + complex_opposite(z)) * n + z * h  # element-wise multiplication
        return h_new


class ComplexBNGRUCell(Module):
    """
    A BN-GRU cell for complex-valued inputs
    """

    def __init__(self, input_length=10, hidden_length=20):
        super().__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # reset gate components
        self.linear_reset_w1 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r1 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.linear_reset_w2 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_reset_r2 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        # update gate components
        self.linear_gate_w3 = ComplexLinear(
            self.input_length, self.hidden_length)
        self.linear_gate_r3 = ComplexLinear(
            self.hidden_length, self.hidden_length)

        self.activation_gate = ComplexSigmoid()
        self.activation_candidate = ComplexTanh()

        self.bn = ComplexBatchNorm2d(1)

    def reset_gate(self, x, h):
        x_1 = self.linear_reset_w1(x)
        h_1 = self.linear_reset_r1(h)
        # gate update
        reset = self.activation_gate(self.bn(x_1) + self.bn(h_1))
        return reset

    def update_gate(self, x, h):
        x_2 = self.linear_reset_w2(x)
        h_2 = self.linear_reset_r2(h)
        z = self.activation_gate(self.bn(h_2) + self.bn(x_2))
        return z

    def update_component(self, x, h, r):
        x_3 = self.linear_gate_w3(x)
        h_3 = r * self.bn(self.linear_gate_r3(h))  # element-wise multiplication
        gate_update = self.activation_candidate(self.bn(self.bn(x_3) + h_3))
        return gate_update

    def forward(self, x, h):
        # Equation 1. reset gate vector
        r = self.reset_gate(x, h)

        # Equation 2: the update gate - the shared update gate vector z
        z = self.update_gate(x, h)

        # Equation 3: The almost output component
        n = self.update_component(x, h, r)

        # Equation 4: the new hidden state
        h_new = (1 + complex_opposite(z)) * n + z * h  # element-wise multiplication
        return h_new


# exists in plain pytorch for complex
class ComplexGRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()

        self.gru_re = GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout,
                          bidirectional=bidirectional)
        self.gru_im = GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, x):
        real, state_real = self._forward_real(x)
        imaginary, state_imag = self._forward_imaginary(x)

        output = torch.complex(real, imaginary)
        state = torch.complex(state_real, state_imag)

        return output, state

    def forward(self, x):
        r2r_out = self.gru_re(x.real)[0]
        r2i_out = self.gru_im(x.real)[0]
        i2r_out = self.gru_re(x.imag)[0]
        i2i_out = self.gru_im(x.imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out

        return torch.complex(real_out, imag_out), None

    def _forward_real(self, x):
        real_real, h_real = self.gru_re(x.real)
        imag_imag, h_imag = self.gru_im(x.imag)
        real = real_real - imag_imag

        return real, torch.complex(h_real, h_imag)

    def _forward_imaginary(self, x):
        imag_real, h_real = self.gru_re(x.imag)
        real_imag, h_imag = self.gru_im(x.real)
        imaginary = imag_real + real_imag

        return imaginary, torch.complex(h_real, h_imag)


# exists in plain pytorch for complex
class ComplexLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.batch_dim = 0 if batch_first else 1
        self.bidirectional = bidirectional

        self.lstm_re = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)
        self.lstm_im = LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, x):
        real, state_real = self._forward_real(x)
        imaginary, state_imag = self._forward_imaginary(x)

        output = torch.complex(real, imaginary)

        return output, (state_real, state_imag)

    def _forward_real(self, x):
        h_real, h_imag, c_real, c_imag = self._init_state(self._get_batch_size(x), x.is_cuda)
        real_real, (h_real, c_real) = self.lstm_re(x.real, (h_real, c_real))
        imag_imag, (h_imag, c_imag) = self.lstm_im(x.imag, (h_imag, c_imag))
        real = real_real - imag_imag
        return real, ((h_real, c_real), (h_imag, c_imag))

    def _forward_imaginary(self, x):
        h_real, h_imag, c_real, c_imag = self._init_state(self._get_batch_size(x), x.is_cuda)
        imag_real, (h_real, c_real) = self.lstm_re(x.imag, (h_real, c_real))
        real_imag, (h_imag, c_imag) = self.lstm_im(x.real, (h_imag, c_imag))
        imaginary = imag_real + real_imag

        return imaginary, ((h_real, c_real), (h_imag, c_imag))

    def _init_state(self, batch_size, to_gpu=False):
        dim_0 = 2 if self.bidirectional else 1
        dims = (dim_0, batch_size, self.hidden_size)

        h_real, h_imag, c_real, c_imag = [
            torch.zeros(dims) for i in range(4)]

        if to_gpu:
            h_real, h_imag, c_real, c_imag = [
                t.cuda() for t in [h_real, h_imag, c_real, c_imag]]

        return h_real, h_imag, c_real, c_imag

    def _get_batch_size(self, x):
        return x.size(self.batch_dim)
