import numpy as np
import torch
from matplotlib import pyplot as plt

from BLL.bayesianlastlayer import BLLModel


def bll_experiment(real_network, complex_network, train_loader, valid_loader, test_loader, output_dim=11, epochs=10,
                   lr=0.0001, ood = False, ood_dataloader=None):
    logs = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "nll": []

    }

    complex_logs = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "nll": []

    }

    complex_model = BLLModel(complex_network, output_dim, {'num_classes': output_dim}, 'cuda', train_loader)
    complex_model.train_model(epochs, train_loader, valid_loader, torch.optim.Adam(complex_model.parameters(), lr=lr),
                              complex_logs)

    real_model = BLLModel(real_network, output_dim, {'num_classes': output_dim}, 'cuda', train_loader)
    real_model.train_model(epochs, train_loader, valid_loader, torch.optim.Adam(real_model.parameters(), lr=lr), logs)

    print("Testing complex model\n")
    complex_prob_matrix = complex_model.test_model(test_loader)
    print("Testing real model\n")
    real_prob_matrix = real_model.test_model(test_loader)

    if ood:
        print("Testing complex model on ood data\n")
        complex_model.test_ood(ood_dataloader)
        print("Testing real model on ood data\n")
        real_model.test_ood(ood_dataloader)

    # fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    # axes = axes.flatten()
    #
    # x = np.arange(output_dim)
    # width = 0.35
    #
    # for i, (real_arr, ax) in enumerate(zip(real_prob_matrix, axes)):
    #     complex_arr = complex_prob_matrix[i]
    #
    #     # colors: orange for the true class, blue for others
    #     real_colors = ['skyblue'] * output_dim
    #     complex_colors = ['lightgreen'] * output_dim
    #     real_colors[i] = 'darkblue'
    #     complex_colors[i] = 'green'
    #
    #     ax.bar(x - width / 2, real_arr, width=width, color=real_colors, alpha=0.5, label="real")
    #     ax.bar(x + width / 2, complex_arr, width=width, color=complex_colors, alpha=0.5, label="complex")
    #
    #     ax.set_ylim(0.00, 1.00)
    #     ax.set_title(f"Class {i} predictions")
    #
    #     ax.legend()
    #
    # axes[-1].axis('off')
    #
    # plt.tight_layout()
    # plt.show()
