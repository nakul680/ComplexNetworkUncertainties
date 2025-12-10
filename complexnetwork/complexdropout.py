import torch
import torch.nn as nn


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Complex Dropout that applies the same mask to both real and imaginary parts

        Args:
            p: dropout probability
        """
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape [batch, 2*channels, ...]
               where channels are organized as [real_ch1, real_ch2, ..., imag_ch1, imag_ch2, ...]
               OR [batch, channels, ..., 2] where last dim is [real, imag]
        """
        if not self.training or self.p == 0:
            return x

        # Determine format
        if x.shape[1] % 2 == 0:  # Format: [batch, 2*channels, ...]
            # Split into real and imaginary
            channels = x.shape[1] // 2
            real = x[:, :channels]
            imag = x[:, channels:]

            # Create mask (same for both real and imag)
            mask = torch.bernoulli(torch.full_like(real, 1 - self.p))
            mask = mask / (1 - self.p)  # Scale to maintain expected value

            # Apply mask
            real = real * mask
            imag = imag * mask

            # Concatenate back
            return torch.cat([real, imag], dim=1)
        else:
            raise ValueError("Input channels must be even for complex dropout")