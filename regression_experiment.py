import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from complexPytorch.complexLoss import NegativeLogLossComplex
from data.function import generate_sinusoidal_data
from deepensemble.ensemble import DeepEnsemble
from realnetwork.toy_regression import LinearRegressionNN

if __name__ == '__main__':

    lag = 20
    train_loader,val_loader, test_loader = generate_sinusoidal_data(15000, lag, 30, 0.02, 0.02)

    ensemble = DeepEnsemble(LinearRegressionNN, 5,{'lag': lag}, task='regression')

    ensemble.train_ensemble(train_loader, val_loader, 30, 0.01, NegativeLogLossComplex())

    ensemble.test_ensemble(test_loader)


