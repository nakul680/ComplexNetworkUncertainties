import torch
from torch.utils.data import Dataset, DataLoader


class RandomOODData(Dataset):
    def __init__(self, num_samples, sample_length, sigma=1.0, device='cpu'):
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.sigma = sigma
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        I = torch.randn(self.sample_length) * self.sigma
        Q = torch.randn(self.sample_length) * self.sigma

        x = torch.stack([I, Q], dim=0)  # (2, L)

        # Dummy label (never used)
        y = -1

        return x.to(self.device), y



