import numpy as np
import torch
from matplotlib import pyplot as plt

from complexnetwork.toy_regression import ComplexRegressionNN
from realnetwork.toy_regression import LinearRegressionNN

def nll_loss(mean, log_var, target):
    """
    Negative log-likelihood for Gaussian distribution
    Loss = 0.5 * (log(var) + (target - mean)^2 / var)
    """
    precision = torch.exp(-log_var)  # 1/variance
    loss = 0.5 * (log_var + precision * (target - mean) ** 2)
    return loss.mean()


def sinusoid_dataset(n_samples=1000, seed=0):
    rng = np.random.RandomState(seed)

    # --- Input domain ---
    x = rng.uniform(-3.0, 3.0, size=(n_samples, 1)).astype(np.float32)

    mu = np.sin(x)

    # --- Input-dependent noise (real-valued std) ---
    sigma = 0.1 * (1.0 + np.exp(-x)) ** -1

    # --- Complex Gaussian noise (independent real/imag parts) ---
    noise = rng.normal(0, sigma, size=(n_samples, 1))


    # --- Observations ---
    y = mu + noise

    return x, y, mu, sigma


if __name__ == '__main__':
    x, y, _, _ = sinusoid_dataset(n_samples=1000)

    # print("mu shape:", mu.shape)
    # print("sigma shape:", sigma.shape)
    #

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # print("x:", x)
    # print("y:", y)

    #model = DeepEnsemble(RegressionNN,1,{'in_features':2}, task="regression")
    model = LinearRegressionNN(1)
    # out = model(Tensor(x))
    # print("out:", out)

    train_dataset = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # model.train_ensemble(train_loader, train_loader, 100, 0.01, ComplexMSE)

    for _ in range(500):
        # loss_fn = torch.nn.MSELoss()
        for x, y in train_loader:
            optimizer.zero_grad()
            # print("x:", x)
            # print("y:", y)
            mean, var = model(x)

            loss = nll_loss(mean, var, y)

            loss.backward()
            print("loss:", loss)
            optimizer.step()

    # Evaluate
    rng = np.random.RandomState(42)
    x_test = rng.uniform(-5.0, 5.0, size=(1000, 1)).astype(np.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        mean, log_var = model(x_test_tensor)
        mean = mean.numpy()
        std = torch.exp(0.5 * log_var).numpy()

    # Sort for better visualization
    sort_idx = x_test.flatten().argsort()
    x_sorted = x_test.flatten()[sort_idx]
    mean_sorted = mean.flatten()[sort_idx]
    std_sorted = std.flatten()[sort_idx]

    # Plot
    plt.figure(figsize=(10, 6))

    x_line = np.linspace(-5, 5, 400)
    plt.plot(x_line, np.sin(x_line), 'k-', linewidth=2, label='True sin(x)', zorder=3)
    plt.plot(x_sorted, mean_sorted, 'b-', linewidth=2, label='Predicted mean', zorder=2)
    plt.fill_between(x_sorted, mean_sorted - 2 * std_sorted, mean_sorted + 2 * std_sorted,
                     alpha=0.3, color='blue', label='±2σ uncertainty', zorder=1)
    plt.scatter(x.numpy(), y.numpy(), s=8, alpha=0.3, c='red', label='Training data', zorder=4)
    plt.axvspan(-3, 3, alpha=0.15, color='green', label='Training region', zorder=0)

    plt.title("Sine Wave Regression", fontsize=14, fontweight='bold')
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()