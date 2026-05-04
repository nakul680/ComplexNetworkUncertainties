import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from complexPytorch.complexLayers import ComplexRNN, RealRegressionNN
from complexPytorch.complexLoss import NegativeLogLossComplex
from complexnetwork.toy_regression import ComplexRegressionNN
from data.function import generate_sinusoidal_data, generate_corr_data, generate_experiment_A
from deepensemble.ensemble import DeepEnsemble
from deepensemblepipeline import ensemble_experiment
from realnetwork.toy_regression import LinearRegressionNN

if __name__ == '__main__':
    lag = 20
    noise = 0.4
    corr = 1.0
    # train_loader, val_loader, test_loader = generate_sinusoidal_data(15000, lag, 30, noise)
    train_loader, val_loader, test_loader = generate_corr_data(15000, lag, 30, noise, corr)
    # train_loader, val_loader, test_loader = generate_experiment_A(15000)

    real_ensemble = DeepEnsemble(LinearRegressionNN, 5, {'lag': lag}, task='regression')
    complex_ensemble = DeepEnsemble(ComplexRegressionNN, 5, {'lag': lag}, task='regression')
    # ensemble.train_ensemble(train_loader, val_loader, 30, 0.01, NegativeLogLossComplex())

    ensemble_experiment(RealRegressionNN, ComplexRegressionNN, train_loader, val_loader, test_loader, lag=lag, task='regression', epochs=10, num_models=5)
