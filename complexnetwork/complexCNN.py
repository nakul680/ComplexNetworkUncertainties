import torch
import torch.nn as nn
import torch.nn.functional as F

from complexPytorch import softmax_real_with_avg, softmax_real_with_abs, softmax_real_with_mult, \
    softmax_of_softmax_real_with_avg, softmax_complex
from complexPytorch.complexLayers import (ComplexBatchNorm1d, ComplexReLU, ComplexLinear,
                                          ComplexDropout, ComplexMaxPool1d)





class ComplexCNN(nn.Module):
    is_complex = True
    def __init__(self, num_classes):
        super(ComplexCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1,dtype=torch.complex64)
        self.bn1 = ComplexBatchNorm1d(64)
        self.pool1 = ComplexMaxPool1d(2)  # 128 -> 64
        self.crelu1 = ComplexReLU()

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1,dtype=torch.complex64)
        self.bn2 = ComplexBatchNorm1d(128)
        self.pool2 = ComplexMaxPool1d(2)  # 64 -> 32
        self.crelu2 = ComplexReLU()

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1,dtype=torch.complex64)
        self.bn3 = ComplexBatchNorm1d(256)
        self.pool3 = ComplexMaxPool1d(2)  # 32 -> 16
        self.crelu3 = ComplexReLU()

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 16, 256,dtype=torch.complex64)
        self.crelu4 = ComplexReLU()
        self.dropout1 = ComplexDropout(0.4)
        self.fc2 = nn.Linear(256, 128,dtype=torch.complex64)
        self.crelu5 = ComplexReLU()
        self.dropout2 = ComplexDropout(0.2)
        self.fc3 = nn.Linear(128, num_classes,dtype=torch.complex64)
        self.real_fc3 = nn.Linear(256, 128)
        self.outlayer = nn.Linear(num_classes*2, num_classes)


    def forward(self, x):
        # x: [batch, 2, 128]
        x = torch.complex(x[:, 0, :], x[:, 1, :])  # [batch, 128]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(self.crelu1(self.bn1(x)))
        x = self.pool2(self.crelu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.crelu3(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = self.crelu4(self.fc1(x))
        x = self.dropout1(x)
        x = self.crelu5(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        # x = softmax_real_with_avg(x, dim=1)
        # x = torch.log(x + 1e-8)
        # x_real = torch.view_as_real(x)
        # x_real = x_real.view(x.size(0) * x_real.size(2), x_real.size(1))
        # x_real = x_real.flatten(start_dim=1)
        # x = self.real_fc3(x_real)
        # x = self.outlayer(x_real)
        # x = x.real
        x = x.abs()
        # x_phase = torch.angle(x)
        # x = torch.cat([x_amp,x_phase], dim=-1)
        # x = x.flatten(start_dim=1)
        # x = self.outlayer(x_real)

        return x

    def compute_features(self, x):
        x = torch.complex(x[:, 0, :], x[:, 1, :])  # [batch, 128]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(self.crelu1(self.bn1(x)))
        x = self.pool2(self.crelu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.crelu3(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = self.crelu4(self.fc1(x))
        x = self.dropout1(x)
        x = self.crelu5(self.fc2(x))
        x = self.dropout2(x)
        x = x.real

        return x



class CVMLP(nn.Module):
    def __init__(self, num_classes=11, hidden_sizes=[512, 256, 128], dropout=0.5):
        super(CVMLP, self).__init__()
        input_size = 128  # flatten 2 channels x 128 length

        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(ComplexLinear(in_features, h))
            layers.append(ComplexReLU())
            layers.append(ComplexDropout(dropout))
            in_features = h
        layers.append(ComplexLinear(in_features, num_classes))  # output layer
        self.outlayer = nn.Linear(num_classes*2, num_classes)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, 2, 128]
        x = torch.complex(x[:, 0, :], x[:, 1, :])
#        x = self.flatten(x)
        x = self.mlp(x)

        x_real = torch.view_as_real(x)
        # x_real = x_real.view(x.size(0) * x_real.size(2), x_real.size(1))
        x_real = x_real.flatten(start_dim=1)
        x = self.outlayer(x_real)
        return x


