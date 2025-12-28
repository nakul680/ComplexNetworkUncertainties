import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from BLL.bayesianlastlayer import BLLModel
from bll_pipeline import bll_experiment
from complexnetwork.complexCNN import ComplexCNN
from deepensemble.ensemble import DeepEnsemble
from deepensemblepipeline import ensemble_experiment
from duq.duq import DUQ
from duq_pipeline import duq_experiment
from realnetwork.amc_cnn import AMC_CNN

if __name__ == "__main__":
    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    snrs = sorted(list(set([key[1] for key in p.keys()])))
    mods = sorted(list(set([key[0] for key in p.keys()])))
    num_classes = len(mods)

    print("Classes:", mods)

    # ---------------------------------------------------
    # 2. BUILD FULL DATASET (X, labels)
    # ---------------------------------------------------
    X_list = []
    y_list = []

    for mod in mods:
        for snr in snrs:
            samples = p[(mod, snr)]  # shape: [N, 2, 128]
            X_list.append(samples)
            y_list += [mods.index(mod)] * samples.shape[0]

    X = np.vstack(X_list)
    Y = np.array(y_list)
    N = len(Y)

    print("Total samples:", N)

    # ---------------------------------------------------
    # 3. TRAIN/VAL/TEST SPLIT (correct indexing)
    # ---------------------------------------------------
    np.random.seed(230983240)

    indices = np.arange(N)
    np.random.shuffle(indices)

    n_train = int(0.6 * N)
    n_valid = int(0.2 * N)
    n_test = N - n_train - n_valid

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    X_train = X[train_idx]
    X_valid = X[valid_idx]
    X_test = X[test_idx]

    Y_train = Y[train_idx]
    Y_valid = Y[valid_idx]
    Y_test = Y[test_idx]

    # ---------------------------------------------------
    # 4. CONVERT TO TORCH TENSORS
    # ---------------------------------------------------
    X_train_tensor = torch.from_numpy(X_train).float()
    X_valid_tensor = torch.from_numpy(X_valid).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    Y_train_tensor = torch.from_numpy(Y_train).long()
    Y_valid_tensor = torch.from_numpy(Y_valid).long()
    Y_test_tensor = torch.from_numpy(Y_test).long()

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=110, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=110, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=110, shuffle=True)

    # bll_experiment(AMC_CNN, ComplexCNN, train_loader, valid_loader, test_loader, len(mods), 10, 0.0001)


    duq_experiment(AMC_CNN, ComplexCNN, train_loader, valid_loader, test_loader, len(mods), 10, 0.0001)


    #ensemble_experiment(AMC_CNN, ComplexCNN, train_loader, valid_loader, test_loader, output_dim=num_classes,num_models=1, epochs=10)


















