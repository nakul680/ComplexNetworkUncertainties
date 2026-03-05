import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import Dataset

from complexnetwork.toy_regression import ComplexRegressionNN
from deepensemble.ensemble import DeepEnsemble
from realnetwork.toy_regression import RegressionNN


def ComplexMSE(x, y):
    mse = nn.MSELoss()
    real_l = mse(x.real, y.real)
    imag_l = mse(x.imag, y.imag)

    return real_l + imag_l


def complex_sinusoid_dataset(n_samples=1000, seed=0):
    rng = np.random.RandomState(seed)

    # --- Input domain ---
    x = rng.uniform(-3.0, 3.0, size=(n_samples, 1)).astype(np.float32)

    # --- Complex mean function ---
    mu = np.sin(x) + 1j * np.cos(x)

    # --- Input-dependent noise (real-valued std) ---
    sigma = 0.15 * (1.0 + np.exp(-x)) ** -1

    # --- Complex Gaussian noise (independent real/imag parts) ---
    noise_real = rng.normal(0, sigma, size=(n_samples, 1))
    noise_imag = rng.normal(0, sigma, size=(n_samples, 1))
    noise = noise_real + 1j * noise_imag

    # --- Observations ---
    y = mu + noise

    return x, y, mu, sigma


# Example usage:
if __name__ == '__main__':
    x, y, _, _ = complex_sinusoid_dataset(n_samples=1000)

    # print("mu shape:", mu.shape)
    # print("sigma shape:", sigma.shape)
    #

    x = torch.tensor(x, dtype=torch.complex64)
    y = torch.tensor(y, dtype=torch.complex64)

    # print("x:", x)
    # print("y:", y)

    #model = DeepEnsemble(RegressionNN,1,{'in_features':2}, task="regression")
    model = ComplexRegressionNN(1)
    # out = model(Tensor(x))
    # print("out:", out)

    train_dataset = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # model.train_ensemble(train_loader, train_loader, 100, 0.01, ComplexMSE)

    for _ in range(100):
        for x, y in train_loader:
            optimizer.zero_grad()
            # print("x:", x)
            # print("y:", y)
            out = model(x)

            loss = ComplexMSE(out, y)

            loss.backward()
            print("loss:", loss)
            optimizer.step()

    rng = np.random.RandomState()
    x_test = rng.uniform(-5.0, 5.0, size=(1000, 1)).astype(np.float32)
    test_out = model.forward(torch.tensor(x_test, dtype=torch.complex64).detach().to('cpu')).detach().numpy()
    #test_out = model(x_test)
    plt.figure(figsize=(10, 5))
    # plt.scatter(x, y.real, s=12, alpha=0.6, label="Re(y)")
    # plt.scatter(x, y.imag, s=12, alpha=0.6, label="Im(y)")

    # Plot true mean functions
    x_line = np.linspace(-4, 4, 400)
    plt.plot(x_line, np.sin(x_line), color="black", linewidth=2, label="Re(mu)=sin(x)")
    plt.plot(x_line, np.cos(x_line), color="gray", linewidth=2, label="Im(mu)=cos(x)")
    plt.scatter(x_test, test_out.real, s=12, alpha=0.6, label="test Re(y)")
    plt.scatter(x_test, test_out.imag, s=12, alpha=0.6, label="test Im(y)")

    plt.title("Complex Sinusoid Dataset (Real & Imag components)")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.show()
