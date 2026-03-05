import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from deepensemble.temp_scaling import TemperatureScaling
from duq.duq import DUQ
from heatmap import fpr_at_95_tpr, calibration_curve, compute_ece, accuracy_rejection_curve
from plot_feature_map import plot_umap, run_umap


def duq_experiment(real_network, complex_network, train_loader, valid_loader, test_loader, output_dim=11, epochs=10,
                   lr=0.0005, feature_dim=128, embed_size=256, length_scale=0.5, ood=False, ood_loader=None):
    complex_temp_scaler = TemperatureScaling()
    real_temp_scaler = TemperatureScaling()
    real_duq = DUQ(
        real_network(output_dim), feature_dim,
        output_dim, embed_size,
        True, length_scale,
        0.999
    )
    real_duq.to('cuda')
    real_duq.train_duq(epochs, train_loader, valid_loader, torch.optim.Adam(real_duq.parameters(), lr=lr), 'cuda', 0.01)


    complex_duq = DUQ(
        complex_network(output_dim), feature_dim,
        output_dim, embed_size,
        True, length_scale,
        0.999
    )
    complex_duq.to('cuda')
    complex_duq.train_duq(epochs, train_loader, valid_loader, torch.optim.Adam(complex_duq.parameters(), lr=lr), 'cuda',
                          0.01)
    # features = []
    # labels = []
    # with torch.no_grad():
    #     for x,y in train_loader:
    #         x = x.float().to('cuda')
    #         y = y.long()
    #
    #         features.append(real_duq.backbone.compute_features(x))
    #         labels.append(y)
    #
    # features = torch.cat(features, dim=0)  # or np.concatenate if numpy
    # labels = torch.cat(labels, dim=0)  # or np.concatenate if numpy
    #
    # # Stratified sampling
    # indices = np.arange(len(labels))
    # _, sampled_indices = train_test_split(
    #     indices,
    #     test_size=0.05,  # Keep 10% of data
    #     stratify=labels.numpy(),
    #     random_state=42
    # )
    #
    # # Use sampled data
    # features = features[sampled_indices]
    # labels = labels[sampled_indices]
    #
    # # Convert to numpy if they're tensors
    # if torch.is_tensor(features):
    #     features = features.detach().cpu().numpy()
    # if torch.is_tensor(labels):
    #     labels = labels.detach().cpu().numpy()
    #
    # embeddings = run_umap(features)
    # embeddings = np.concatenate(embeddings, axis=0)
    #
    # if isinstance(embeddings, list):
    #     embeddings = np.array(embeddings)
    #
    # plt.figure(figsize=(8, 6))
    #
    # # Plot each class with different colors
    # classes = np.unique(labels)
    # colors = ['blue', 'red', 'green', 'orange', 'purple','yellow','black','brown','pink','cyan', 'magenta']
    #
    # for i, cls in enumerate(classes):
    #     mask = labels == cls
    #     plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
    #                 c=colors[i],
    #                 label=f'Class {cls}',
    #                 alpha=0.6,
    #                 s=50,
    #                 edgecolors='black',
    #                 linewidths=0.5)
    #
    # plt.xlabel('Feature 1', fontsize=12)
    # plt.ylabel('Feature 2', fontsize=12)
    # plt.title('Data Scatter Plot by Class', fontsize=14, fontweight='bold')
    # # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    #
    #
    #


    print("Testing real network")
    real_temp_scaler.fit(real_duq, valid_loader)
    real_id_likelihood, real_correct_likelihood, real_wrong_likelihood, real_preds, real_labels = real_duq.test_duq(test_loader, 'cuda')
    print("Testing complex network")
    complex_temp_scaler.fit(complex_duq, valid_loader)
    complex_id_likelihood, complex_correct_likelihood, complex_wrong_likelihood, complex_preds, complex_labels = complex_duq.test_duq(test_loader, 'cuda')

    if ood:
        complex_ood_likelihood, _, _, _, _ = complex_duq.test_duq(ood_loader, 'cuda')
        real_ood_likelihood, _, _, _, _ = real_duq.test_duq(ood_loader, 'cuda')

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
        print(f"Complex Model FPR@95TPR: {fpr_at_95_tpr(complex_fpr, complex_tpr)}")
        print(f"Real Model FPR@95TPR: {fpr_at_95_tpr(real_fpr, real_tpr)}")

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

    print(f"Complex ECE:{compute_ece(complex_id_likelihood, complex_preds, complex_labels)}")
    print(f"Real ECE:{compute_ece(real_id_likelihood, real_preds, real_labels)}")
    c_conf, c_acc, c_count = calibration_curve(
        complex_id_likelihood,
        complex_preds.numpy(),
        complex_labels.numpy()
    )

    # Real model calibration
    r_conf, r_acc, r_count = calibration_curve(
        real_id_likelihood,
        real_preds.numpy(),
        real_labels.numpy()
    )

    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.plot(c_conf, c_acc, marker='o', label='Complex')
    plt.plot(r_conf, r_acc, marker='o', label='Real')
    plt.legend()
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Comparison')
    plt.grid(True)
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

    fig, ax = plt.subplots(1,2,figsize=(10, 6))

    ax[0].hist(real_correct_likelihood, bins='auto', alpha=0.5, label='Real Network correct predictions')
    ax[0].hist(real_wrong_likelihood, bins='auto', alpha=0.5, label='Real Network wrong predictions')

    ax[1].hist(complex_correct_likelihood, bins='auto', alpha=0.5, label='Complex Network correct predictions')
    ax[1].hist(complex_wrong_likelihood, bins='auto', alpha=0.5, label='Complex Network wrong predictions')

    ax[0].legend()
    ax[1].legend()
    plt.show()

    real_accuracy, real_rejection = accuracy_rejection_curve(real_labels.detach().numpy(), real_preds.detach().numpy(), real_id_likelihood)
    complex_accuracy, complex_rejection = accuracy_rejection_curve(complex_labels.detach().numpy(), complex_preds.detach().numpy(), complex_id_likelihood)

    plt.plot(real_rejection, real_accuracy, label=" Real Accuracy-Rejection Curve")
    plt.plot(complex_rejection, complex_accuracy, label=" Complex Accuracy-Rejection Curve")
    plt.xlabel("Rejection Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy-Rejection Curve")
    plt.grid(True)
    plt.legend()
    plt.show()





def extract_duq_features(duq_model, loader, device):
    duq_model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = duq_model.compute_features(x)  # penultimate features
            features.append(z.cpu().numpy())
            labels.append(y.numpy())

    return np.concatenate(features), np.concatenate(labels)


def compute_feature_centroids(features, labels, num_classes):
    centroids = []
    for c in range(num_classes):
        centroids.append(features[labels == c].mean(axis=0))
    return np.vstack(centroids)