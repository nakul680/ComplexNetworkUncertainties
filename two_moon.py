import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from sklearn.datasets import make_moons
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from complexnetwork.complexCNN import CVMLP_RLL, CVMLP_ABS
from deepensemble.ensemble import DeepEnsemble
from duq.duq import DUQ
from heatmap import count_params
from realnetwork.amc_cnn import AMC_CNN, AMC_MLP


def two_moon_experiment(model, ax):
    print(count_params(model))
    X, y = make_moons(n_samples=1000, noise=0.2)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    if model.backbone.is_complex:
        model_type = 'complex'
    else:
        model_type = 'real'

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=100, shuffle=True)

    # real_model = CVMLP(2)
    # real_model = DeepEnsemble(AMC_MLP,5,{'num_classes': 2}, device='cpu')
    # model.train_ensemble(train_loader,test_loader,10,0.003)
    #real_model = DUQ(CVMLP(2),128,2,embedding_size=64,learnable_length_scale=True,length_scale=0.05)
    model.train_duq(30, train_loader, test_loader, torch.optim.Adam(model.parameters(), lr=0.001), 'cpu',lgp=1.0)
    # optimizer = torch.optim.Adam(real_model.parameters(), lr=0.001)
    # loss_fn = torch.nn.CrossEntropyLoss()

    # real_model.train()
    # for _ in range(300):
    #     for X, y in dataloader:
    #         optimizer.zero_grad()
    #         X = torch.tensor(X, dtype=torch.float32)
    #         y = torch.tensor(y, dtype=torch.long)
    #
    #         out = real_model(X)
    #
    #         loss = loss_fn(out, y)
    #         loss.backward()
    #         optimizer.step()

    xx, yy = np.meshgrid(
        np.linspace(-2, 3, 300),
        np.linspace(-2, 2, 300)
    )
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    )

    # model.eval()
    probs = model.forward(grid).detach().numpy()
    # probs = torch.softmax(logits, dim=1)
    # probs = probs.detach().cpu().numpy()
    # uncertainty = -np.sum(
    #     probs * np.log(probs + 1e-8),
    #     axis=1
    # )
    uncertainty = -np.max(probs, axis=1)
    print(len(uncertainty))

    u = uncertainty.reshape(xx.shape)
    # plt.figure(figsize=(6, 5))

    # Option 1: Power-law normalization for enhanced sensitivity
    cf = ax.contourf(xx, yy, u, levels=50, cmap="viridis",
                 norm=PowerNorm(gamma=0.1))  # gamma < 1 emphasizes small differences

    # ax.colorbar(label="Uncertainty")

    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="coolwarm",
        edgecolor="black",
        s=25
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Uncertainty Grid: {model_type}")
    # plt.show()
    return cf



if __name__ == "__main__":
    fig, ax = plt.subplots(1,3, figsize=(18,5))
    model_rll = DUQ(CVMLP_RLL(2), 128, 2, embedding_size=10, learnable_length_scale=True, length_scale=0.3)
    # model = DeepEnsemble(AMC_MLP, 5, {'num_classes': 2},'cpu')

    # plot_rll.show()
    model_abs = DUQ(CVMLP_ABS(2), 128, 2, embedding_size=10, learnable_length_scale=True, length_scale=0.3)
    model_real = DUQ(AMC_MLP(2), 128, 2, embedding_size=10, learnable_length_scale=True, length_scale=0.3)
    cf1 = two_moon_experiment(model_abs,ax[1])
    cf0 = two_moon_experiment(model_real, ax[0])
    cf2 = two_moon_experiment(model_rll, ax[2])

   # fig.colorbar(cf2, ax=ax, label="Uncertainty")

    # ax[0].plot(plot_real)
    # ax[1].plot(plot_abs)
    # ax[2].plot(plot_rll)
    plt.tight_layout()
    plt.show()


