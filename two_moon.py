import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from sklearn.datasets import make_moons
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from complexnetwork.complexCNN import CVMLP_RLL, CVMLP_ABS
from deepensemble.ensemble import DeepEnsemble
from deepensemble.uncertainty import compute_and_plot_auroc
from duq.duq import DUQ
from heatmap import count_params, compute_ece
from realnetwork.amc_cnn import AMC_CNN, AMC_MLP


def two_moon_experiment(model, ax, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    print(f"Model params: {count_params(model)}")
    X, y = make_moons(n_samples=5000, noise=0.5)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=100, shuffle=True)

    model.train_duq(30, train_loader, test_loader, torch.optim.Adam(model.parameters(), lr=0.001), 'cpu', lgp=1.0)

    # Test on test set
    model.eval()
    with torch.no_grad():
        scores = model(X_test)
        probs = torch.softmax(scores, dim=1)
        preds = torch.argmax(scores, dim=1)
        accuracy = (preds == y_test).float().mean().item()

    auroc = roc_auc_score(y_test.numpy(), scores[:,1].numpy())
    ece = compute_ece(scores, preds, y_test.numpy())

    xx, yy = np.meshgrid(
        np.linspace(-3, 4, 300),
        np.linspace(-2, 3, 300)
    )
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        scores = model(grid).numpy()
        # probs = torch.softmax(logits, dim=1).numpy()

    # auroc = compute_and_plot_auroc(id_probs.max(1), probs.max(1),)
    uncertainty = -np.max(scores, axis=1)
    u = uncertainty.reshape(xx.shape)

    if ax is not None:
        cf = ax.contourf(xx, yy, u, levels=50, cmap="viridis",
                         norm=PowerNorm(gamma=0.5))

        idx = np.random.choice(len(X_train), size=int(0.4 * len(X_train)), replace=False)
        ax.scatter(
            X_train[idx, 0],
            X_train[idx, 1],
            c=y_train[idx],
            cmap="coolwarm",
            edgecolor="black",
            s=25,
            alpha=0.1,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Uncertainty Grid: {model.backbone.model_name}")

        return cf, accuracy, ece
    return None, accuracy, ece


def run_experiment(backbone_class, backbone_kwargs, num_runs=5):
    """Run experiment multiple times with different seeds."""
    accs = []
    eces = []
    for seed in range(num_runs):
        model = DUQ(backbone_class(**backbone_kwargs), 128, 2,
                    embedding_size=10, learnable_length_scale=True, length_scale=0.3)
        _, acc, ece = two_moon_experiment(model, None, seed=seed)
        accs.append(acc)
        eces.append(ece)

    return np.mean(accs), np.std(accs), np.mean(eces), np.std(eces)


if __name__ == "__main__":
    num_runs = 5
    print(f"Running {num_runs} seeds per model...\n")

    # Run multiple times to get stats
    acc_real_mean, acc_real_std,ece_real_mean, ece_real_std = run_experiment(AMC_MLP, {'num_classes': 2}, num_runs=num_runs)
    acc_abs_mean, acc_abs_std, ece_abs_mean,ece_abs_std = run_experiment(CVMLP_ABS, {'num_classes': 2}, num_runs=num_runs)
    acc_rll_mean, acc_rll_std, ece_rll_mean,ece_rll_std = run_experiment(CVMLP_RLL, {'num_classes': 2}, num_runs=num_runs)

    # Print summary
    print("\n" + "=" * 50)
    print("Test Accuracy Summary (mean ± std):")
    print("=" * 50)
    print(f"RVMLP:      {acc_real_mean:.4f} ± {acc_real_std:.4f}")
    print(f"CVMLP-ABS:  {acc_abs_mean:.4f} ± {acc_abs_std:.4f}")
    print(f"CVMLP-RLL:  {acc_rll_mean:.4f} ± {acc_rll_std:.4f}")
    print("=" * 50 + "\n")



    # Create visualization with best seed (seed=0 or highest accuracy)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    model_real = DUQ(AMC_MLP(num_classes=2), 128, 2, embedding_size=10, learnable_length_scale=True, length_scale=0.3)
    model_abs = DUQ(CVMLP_ABS(num_classes=2), 128, 2, embedding_size=10, learnable_length_scale=True, length_scale=0.3)
    model_rll = DUQ(CVMLP_RLL(num_classes=2), 128, 2, embedding_size=10, learnable_length_scale=True, length_scale=0.3)

    cf0, _,_ = two_moon_experiment(model_real, ax[0], seed=0)
    cf1, _,_ = two_moon_experiment(model_abs, ax[1], seed=0)
    cf2, _,_ = two_moon_experiment(model_rll, ax[2], seed=0)

    # Print summary
    print("\n" + "=" * 50)
    print("Test Accuracy Summary (mean ± std):")
    print("=" * 50)
    print(f"RVMLP:      {acc_real_mean:.4f} ± {acc_real_std:.4f}")
    print(f"CVMLP-ABS:  {acc_abs_mean:.4f} ± {acc_abs_std:.4f}")
    print(f"CVMLP-RLL:  {acc_rll_mean:.4f} ± {acc_rll_std:.4f}")
    print("=" * 50 + "\n")

    # Print summary
    print("\n" + "=" * 50)
    print("Test ECE Summary (mean ± std):")
    print("=" * 50)
    print(f"RVMLP:      {ece_real_mean:.4f} ± {ece_real_std:.4f}")
    print(f"CVMLP-ABS:  {ece_abs_mean:.4f} ± {ece_abs_std:.4f}")
    print(f"CVMLP-RLL:  {ece_rll_mean:.4f} ± {ece_rll_std:.4f}")
    print("=" * 50 + "\n")

    plt.tight_layout()
    fig.colorbar(cf0, ax=ax, shrink=0.9)
    plt.show()
