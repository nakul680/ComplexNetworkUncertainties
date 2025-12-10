import numpy as np
import torch
from matplotlib import pyplot as plt

from duq.duq import DUQ


def duq_experiment(real_network, complex_network, train_loader, valid_loader, test_loader, output_dim=11, epochs=10,
                   lr=0.0005, feature_dim=128, embed_size=256, length_scale=0.5):
    real_duq = DUQ(
        real_network(output_dim), feature_dim,
        output_dim, embed_size,
        True, length_scale,
        0.999
    )
    real_duq.to('cuda')
    real_duq.train_duq(10, train_loader, valid_loader, torch.optim.Adam(real_duq.parameters(), lr=lr), 'cuda', 0.00)
    real_max_distances, real_correct, real_wrong = real_duq.test_duq(test_loader, 'cuda')

    complex_duq = DUQ(
        complex_network(output_dim), feature_dim,
        output_dim, embed_size,
        True, length_scale,
        0.999
    )
    complex_duq.to('cuda')
    complex_duq.train_duq(10, train_loader, valid_loader, torch.optim.Adam(complex_duq.parameters(), lr=lr), 'cuda',
                          0.00)
    complex_max_distances, complex_correct, complex_wrong = complex_duq.test_duq(test_loader, 'cuda')

    categories = ['Uncertainty on Correct Predictions', 'Uncertainty on Wrong Predictions']
    set1_avgs = [1 - sum(real_correct)/len(real_correct), 1 - sum(real_wrong)/len(real_wrong)]
    set2_avgs = [1 - sum(complex_correct)/len(complex_correct), 1 - sum(complex_wrong)/len(complex_wrong)]
    x = np.arange(len(categories))
    width = 0.35

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars1 = ax.bar(x - width / 2, set1_avgs, width, label="Real network", alpha=0.8)
    bars2 = ax.bar(x + width / 2, set2_avgs, width, label="Complex network", alpha=0.8)

    # Customize plot
    ax.set_xlabel('Prediction Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Max Distance', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Average Max Distances\nfor Correct vs Wrong Predictions',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.legend()
    plt.show()
