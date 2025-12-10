import numpy as np
import torch
from matplotlib import pyplot as plt

from deepensemble.ensemble import DeepEnsemble


def ensemble_experiment(real_network, complex_network,train_loader, val_loader, test_loader,output_dim, num_models=5, device='cuda',epochs=10):
    real_ensemble = DeepEnsemble(real_network, num_models, {'num_classes': output_dim}, device)
    real_ensemble.train_ensemble(train_loader,val_loader,epochs)

    complex_ensemble = DeepEnsemble(complex_network, num_models, {'num_classes': output_dim}, device)
    complex_ensemble.train_ensemble(train_loader, val_loader, epochs)

    print("Testing complex ensemble...")
    complex_metrics = complex_ensemble.test_ensemble(test_loader, torch.nn.CrossEntropyLoss())
    print("Testing real ensemble...")
    real_metrics = real_ensemble.test_ensemble(test_loader, torch.nn.CrossEntropyLoss())

    categories = ['Uncertainty on Correct Predictions', 'Uncertainty on Wrong Predictions']
    set1_avgs = [real_metrics['avg_correct_entropy'], complex_metrics['avg_correct_entropy']]
    set2_avgs = [real_metrics['avg_wrong_entropy'], complex_metrics['avg_wrong_entropy']]
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


