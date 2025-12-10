import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_class_prob_heatmap(model, dataloader, num_classes=11, device='cuda'):
    """
    Generates an 11x11 heatmap of average predicted probabilities.

    Rows  = true labels (0..10)
    Cols  = predicted class index (0..10)
    Cell (i,j) = average prob assigned to class j for all samples with true label i.
    """
    # accumulator: rows = true labels, cols = predicted classes
    prob_matrix = np.zeros((num_classes, num_classes))
    count_per_label = np.zeros(num_classes)

    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            labels = y.tolist()

            for label in labels:
                count_per_label[label] += 1

            out = model(x)
            probs = out.predictive.probs  # shape [B, 11]

            # for lbl in range(num_classes):
            #     mask = (y == lbl)
            #     if mask.sum() == 0:
            #         continue

            # selected_probs = probs[mask]  # shape [N_lbl, 11]
            # prob_matrix[lbl] += selected_probs.sum(dim=0).cpu().numpy()
            # count_per_label[lbl] += mask.sum().item()

            for i in range(len(labels)):
                prob_matrix[labels[i]] += probs[i].tolist()

    # normalize rows by how many batches contributed
    for i in range(num_classes):
        if count_per_label[i] > 0:
            prob_matrix[i] /= count_per_label[i]

    # --- plot heatmap ---
    plt.figure(figsize=(10, 8))
    plt.imshow(prob_matrix, cmap='magma', aspect='auto', vmin=0.00, vmax=0.70)
    plt.colorbar(label="Average predicted probability")

    plt.xticks(ticks=np.arange(num_classes), labels=np.arange(num_classes))
    plt.yticks(ticks=np.arange(num_classes), labels=np.arange(num_classes))

    plt.xlabel("Predicted Class")
    plt.ylabel("True Label")
    plt.title("11-Class Prediction Probability Heatmap")
    plt.tight_layout()
    plt.show()

    return prob_matrix
