import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from complexPytorch import softmax_real_with_avg
from heatmap import compute_ece


def calculate_classification_uncertainty(ensemble_predictions, true_labels, batch_size=32, ood_label=100):
    """
    Calculate uncertainty metrics for classification, with batching and batch-wise value plots.
    Supports OOD (out-of-distribution) samples.

    Args:
        ensemble_predictions: np.ndarray, shape (num_models, sample_size, num_classes)
                             Logits from ensemble models (NOT probabilities)
        true_labels: np.ndarray, shape (sample_size,) - integer class labels
                     OOD samples should have labels >= num_classes (e.g., 100, 1000)
        batch_size: int, number of samples per batch
        ood_label: int or None, specific label value indicating OOD samples
                   If None, any label >= num_classes is treated as OOD

    Returns:
        dict with:
            - overall metrics (nll, accuracy, brier_score, predictive_entropy)
            - per-sample values
            - per-batch NLL and Brier score arrays
            - separate metrics for in-distribution and OOD samples
    """
    epsilon = 1e-10
    num_classes = ensemble_predictions.shape[2]
    sample_size = ensemble_predictions.shape[1]

    ensemble_logits = torch.tensor(ensemble_predictions)
    true_labels = torch.tensor(true_labels).long()

    M, N, C = ensemble_logits.shape

    # --------------------------------------------------
    # 0. Convert logits -> probs
    # --------------------------------------------------
    if ensemble_logits.dtype == torch.complex64:
        probs = softmax_real_with_avg(ensemble_logits, dim = -1)
        log_probs = torch.log(probs + epsilon)
    else:
        log_probs = F.log_softmax(ensemble_logits, dim=-1)  # (M, N, C)
        probs = log_probs.exp()  # (M, N, C)

    # --------------------------------------------------
    # 1. Predictive distribution (average across ensemble)
    # --------------------------------------------------
    mean_probs = probs.mean(dim=0)  # (N, C)

    # --------------------------------------------------
    # 2. Identify OOD samples
    # --------------------------------------------------
    if ood_label is not None:
        ood_mask = true_labels == ood_label
    else:
        ood_mask = true_labels >= num_classes

    id_mask = ~ood_mask

    # --------------------------------------------------
    # 3. Accuracy (only for in-distribution samples)
    # --------------------------------------------------
    predicted_classes = mean_probs.argmax(dim=1)

    if id_mask.any():
        accuracy = (predicted_classes[id_mask] == true_labels[id_mask]).float().mean().item()
        ece = compute_ece(mean_probs[id_mask], predicted_classes[id_mask], true_labels[id_mask])
    else:
        accuracy = float('nan')
        ece = float('nan')

    # --------------------------------------------------
    # 4. NLL (only for in-distribution samples)
    # --------------------------------------------------
    nll_per_sample = torch.full((N,), float('nan'))

    if id_mask.any():
        id_indices = torch.where(id_mask)[0]
        true_class_probs = mean_probs[id_indices, true_labels[id_indices]]
        nll_per_sample[id_indices] = -torch.log(true_class_probs + epsilon)


    avg_nll = nll_per_sample[id_mask].mean().item() if id_mask.any() else float('nan')

    # --------------------------------------------------
    # 5. Brier score (only for in-distribution samples)
    # --------------------------------------------------
    brier_per_sample = torch.full((N,), float('nan'))

    if id_mask.any():
        id_indices = torch.where(id_mask)[0]
        t_star = torch.zeros((id_mask.sum(), C))
        t_star[torch.arange(id_mask.sum()), true_labels[id_indices]] = 1.0
        brier_per_sample[id_indices] = torch.sum((t_star - mean_probs[id_indices]) ** 2, dim=1)

    brier_score = brier_per_sample[id_mask].mean().item() if id_mask.any() else float('nan')

    # --------------------------------------------------
    # 6. Predictive entropy (calculated for all samples)
    # --------------------------------------------------
    predictive_entropy = -torch.sum(
        mean_probs * torch.log(mean_probs + epsilon), dim=1
    )
    avg_predictive_entropy = predictive_entropy.mean().item()
    avg_predictive_entropy_id = predictive_entropy[id_mask].mean().item() if id_mask.any() else float('nan')
    avg_predictive_entropy_ood = predictive_entropy[ood_mask].mean().item() if ood_mask.any() else float('nan')

    # --------------------------------------------------
    # 7. Expected entropy (aleatoric) - for all samples
    # --------------------------------------------------
    expected_entropy = -torch.mean(
        torch.sum(probs * log_probs, dim=2), dim=0
    )  # (N,)
    avg_expected_entropy = expected_entropy.mean().item()
    avg_expected_entropy_id = expected_entropy[id_mask].mean().item() if id_mask.any() else float('nan')
    avg_expected_entropy_ood = expected_entropy[ood_mask].mean().item() if ood_mask.any() else float('nan')

    # --------------------------------------------------
    # 8. Mutual information (epistemic) - for all samples
    # --------------------------------------------------
    mutual_information = predictive_entropy - expected_entropy
    avg_mutual_information = mutual_information.mean().item()
    avg_mutual_information_id = mutual_information[id_mask].mean().item() if id_mask.any() else float('nan')
    avg_mutual_information_ood = mutual_information[ood_mask].mean().item() if ood_mask.any() else float('nan')

    # --------------------------------------------------
    # 9. Prediction variance - for all samples
    # --------------------------------------------------
    prediction_variance = torch.var(probs, dim=0).mean(dim=1)  # (N,)
    avg_prediction_variance = prediction_variance.mean().item()
    avg_prediction_variance_id = prediction_variance[id_mask].mean().item() if id_mask.any() else float('nan')
    avg_prediction_variance_ood = prediction_variance[ood_mask].mean().item() if ood_mask.any() else float('nan')

    # --------------------------------------------------
    # 10. Max probability (confidence) - for all samples
    # --------------------------------------------------
    max_probs = mean_probs.max(dim=1)[0]
    avg_max_prob = max_probs.mean().item()
    max_probs_id = max_probs[id_mask]
    avg_max_prob_id = max_probs[id_mask].mean().item() if id_mask.any() else float('nan')
    avg_max_prob_ood = max_probs[ood_mask].mean().item() if ood_mask.any() else float('nan')

    # --------------------------------------------------
    # 11. Correct vs incorrect metrics (only for ID samples)
    # --------------------------------------------------
    correct_mask = (predicted_classes == true_labels) & id_mask
    incorrect_mask = (predicted_classes != true_labels) & id_mask

    avg_entropy_correct = (
        predictive_entropy[correct_mask].mean().item()
        if correct_mask.any()
        else float("nan")
    )
    avg_entropy_incorrect = (
        predictive_entropy[incorrect_mask].mean().item()
        if incorrect_mask.any()
        else float("nan")
    )

    avg_max_prob_correct = (
        max_probs[correct_mask].mean().item()
        if correct_mask.any()
        else float("nan")
    )
    avg_max_prob_incorrect = (
        max_probs[incorrect_mask].mean().item()
        if incorrect_mask.any()
        else float("nan")
    )

    # --------------------------------------------------
    # 12. Batch-wise metrics (only for ID samples)
    # --------------------------------------------------
    num_batches = math.ceil(N / batch_size)
    nll_batches = []
    brier_batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, N)
        batch_id_mask = id_mask[start:end]

        if batch_id_mask.any():
            nll_batches.append(nll_per_sample[start:end][batch_id_mask].mean().item())
            brier_batches.append(brier_per_sample[start:end][batch_id_mask].mean().item())
        else:
            nll_batches.append(float('nan'))
            brier_batches.append(float('nan'))



    return {
        # Overall metrics (ID samples only for supervised metrics)
        "accuracy": accuracy,
        "nll": avg_nll,
        "brier_score": brier_score,

        # Uncertainty metrics (all samples)
        "predictive_entropy": avg_predictive_entropy,
        "expected_entropy": avg_expected_entropy,
        "mutual_information": avg_mutual_information,
        "avg_prediction_variance": avg_prediction_variance,
        "avg_max_prob": avg_max_prob,
        "ece": ece,
        "max_probs": max_probs,
        "predictions": predicted_classes,
        "labels": true_labels,

        # ID vs OOD breakdown
        "num_id_samples": id_mask.sum().item(),
        "num_ood_samples": ood_mask.sum().item(),

        "predictive_entropy_id": avg_predictive_entropy_id,
        "predictive_entropy_ood": avg_predictive_entropy_ood,

        "expected_entropy_id": avg_expected_entropy_id,
        "expected_entropy_ood": avg_expected_entropy_ood,

        "mi_scores": mutual_information,
        "mutual_information_id": avg_mutual_information_id,
        "mutual_information_ood": avg_mutual_information_ood,

        "prediction_variance_id": avg_prediction_variance_id,
        "prediction_variance_ood": avg_prediction_variance_ood,

        "max_prob_id": avg_max_prob_id,
        "max_prob_ood": avg_max_prob_ood,

        # Correct vs incorrect (ID only)
        "avg_correct_entropy": avg_entropy_correct,
        "avg_wrong_entropy": avg_entropy_incorrect,
        "avg_correct_max_prob": avg_max_prob_correct,
        "avg_wrong_max_prob": avg_max_prob_incorrect,

        # Batch-wise metrics
        "nll_per_batch": nll_batches,
        "brier_per_batch": brier_batches,
    }


def compute_and_plot_auroc(id_scores, ood_scores, label, ax=None):
    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])

    scores = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, scores)

    fpr, tpr, _ = roc_curve(y_true, scores)

    if ax is not None:
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auroc:.3f})")

    return auroc
