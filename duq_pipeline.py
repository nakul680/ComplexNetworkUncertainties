import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from duq.duq import DUQ


def duq_experiment(real_network, complex_network, train_loader, valid_loader, test_loader, output_dim=11, epochs=10,
                   lr=0.0005, feature_dim=128, embed_size=256, length_scale=0.5, ood=False, ood_loader=None):
    real_duq = DUQ(
        real_network(output_dim), feature_dim,
        output_dim, embed_size,
        True, length_scale,
        0.999
    )
    real_duq.to('cuda')
    real_duq.train_duq(epochs, train_loader, valid_loader, torch.optim.Adam(real_duq.parameters(), lr=lr), 'cuda', 0.00)


    complex_duq = DUQ(
        complex_network(output_dim), feature_dim,
        output_dim, embed_size,
        True, length_scale,
        0.999
    )
    complex_duq.to('cuda')
    complex_duq.train_duq(epochs, train_loader, valid_loader, torch.optim.Adam(complex_duq.parameters(), lr=lr), 'cuda',
                          0.00)

    real_id_likelihood, real_correct_likelihood, real_wrong_likelihood = real_duq.test_duq(test_loader, 'cuda')
    complex_id_likelihood, complex_correct_likelihood, complex_wrong_likelihood = complex_duq.test_duq(test_loader, 'cuda')

    if ood:
        complex_ood_likelihood, _, _ = complex_duq.test_duq(ood_loader, 'cuda')
        real_ood_likelihood, _, _ = real_duq.test_duq(ood_loader, 'cuda')

        print(f"Complex OOD uncertainty: {1 - np.mean(complex_ood_likelihood)}")
        print(f"Real OOD uncertainty: {1 - np.mean(real_ood_likelihood)}")

        real_id_scores = 1 - np.array(real_id_likelihood)
        real_ood_scores = 1 - np.array(real_ood_likelihood)

        complex_id_scores = 1 - np.array(complex_id_likelihood)
        complex_ood_scores = 1 - np.array(complex_ood_likelihood)

        # --- Labels ---
        id_labels = np.zeros(len(real_id_scores))
        ood_labels = np.ones(len(real_ood_scores))

        labels = np.concatenate([id_labels, ood_labels])

        # --- Real network AUROC ---
        real_scores = np.concatenate([real_id_scores, real_ood_scores])
        real_auroc = roc_auc_score(labels, real_scores)

        # --- Complex network AUROC ---
        complex_scores = np.concatenate([complex_id_scores, complex_ood_scores])
        complex_auroc = roc_auc_score(labels, complex_scores)

        print(f"Real AUROC: {real_auroc:.4f}")
        print(f"Complex AUROC: {complex_auroc:.4f}")

        # --- ROC curves ---
        real_fpr, real_tpr, _ = roc_curve(labels, real_scores)
        complex_fpr, complex_tpr, _ = roc_curve(labels, complex_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(real_fpr, real_tpr, label=f"Real (AUROC = {real_auroc:.3f})")
        plt.plot(complex_fpr, complex_tpr, label=f"Complex (AUROC = {complex_auroc:.3f})")
        plt.plot([0, 1], [0, 1], '--', color='gray', label="Random")

        plt.xlabel("False Positive Rate (ID → OOD)")
        plt.ylabel("True Positive Rate (OOD detected)")
        plt.title("OOD Detection ROC Curve (DUQ)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    categories = ['Uncertainty on Correct Predictions', 'Uncertainty on Wrong Predictions']
    set1_avgs = [1 - (sum(real_correct_likelihood) / len(real_correct_likelihood)), 1 - (sum(real_wrong_likelihood) / len(real_wrong_likelihood))]
    set2_avgs = [1 - (sum(complex_correct_likelihood) / len(complex_correct_likelihood)), 1 - (sum(complex_wrong_likelihood) / len(complex_wrong_likelihood))]
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
