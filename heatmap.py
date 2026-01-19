import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


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



def fpr_at_95_tpr(fpr, tpr):
    """
    Compute FPR@95%TPR given ROC curve values.

    Args:
        fpr (np.ndarray): false positive rates
        tpr (np.ndarray): true positive rates

    Returns:
        float: FPR at 95% TPR
    """

    idx = np.where(tpr >= 0.95)[0]

    if len(idx) == 0:
        return 1.0  # worst case

    return fpr[idx[0]]


def calibration_curve(confidences, predictions, labels, n_bins=10):
    confidences = np.asarray(confidences)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_count = []

    correct = (predictions == labels)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if np.sum(mask) > 0:
            bin_acc.append(np.mean(correct[mask]))
            bin_conf.append(np.mean(confidences[mask]))
            bin_count.append(np.sum(mask))

    return np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


def plot_calibration(conf, acc, name):
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    plt.plot(conf, acc, marker='o', label=name)
    plt.xlabel('Predicted confidence')
    plt.ylabel('Empirical accuracy')
    plt.title(f'Calibration Plot: {name}')
    plt.legend()
    plt.grid(True)
    plt.show()



def compute_ece(probs, preds, labels, n_bins=15):
    probs = np.asarray(probs)
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    # Use max softmax probability as confidence
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
    else:
        confidences = probs  # already max-prob

    accuracies = (preds == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidences)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(accuracies[mask])
            ece += (np.sum(mask) / N) * abs(bin_acc - bin_conf)

    return ece


def count_params(model):
    real_equiv_params = 0
    for p in model.parameters():
        if p.is_complex():
            real_equiv_params += 2 * p.numel()
        else:
            real_equiv_params += p.numel()

    return real_equiv_params
