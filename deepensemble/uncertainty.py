import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from complexPytorch import softmax_real_with_avg
from complexPytorch.complexLoss import NegativeLogLossComplex
from heatmap import compute_ece


def calculate_classification_uncertainty(ensemble_predictions, true_labels, batch_size=100, ood_label=100):
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
        probs = softmax_real_with_avg(ensemble_logits, dim=-1)
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
        "all_entropy": predictive_entropy,
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

        "prediction_variance": prediction_variance,
        "prediction_variance_id": avg_prediction_variance_id,
        "prediction_variance_ood": avg_prediction_variance_ood,

        "max_prob_id": avg_max_prob_id,
        "max_prob_ood": avg_max_prob_ood,

        # Correct vs incorrect (ID only)
        "avg_correct_entropy": avg_entropy_correct,
        "avg_wrong_entropy": avg_entropy_incorrect,
        "correct_max_probs": max_probs[correct_mask],
        "wrong_max_probs": max_probs[incorrect_mask],
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


def calculate_regression_uncertainty(ensemble_means, ensemble_var_reals, ensemble_var_imags, ensemble_cov, true_results):
    """
    Args:
        ensemble_means     : (n_models, n_samples) complex
        ensemble_var_reals : (n_models, n_samples) real
        ensemble_var_imags : (n_models, n_samples) real
        true_results       : (n_samples,) complex
    """
    # --- setup ---
    means = torch.tensor(ensemble_means, dtype=torch.complex64)  # (n_models, n_samples)
    var_reals = torch.tensor(ensemble_var_reals, dtype=torch.float32)  # (n_models, n_samples)
    var_imags = torch.tensor(ensemble_var_imags, dtype=torch.float32)  # (n_models, n_samples)
    cov = torch.tensor(ensemble_cov, dtype=torch.float32)
    y = torch.tensor(true_results, dtype=torch.complex64)  # (n_samples,)

    mean_real = means.real  # (n_models, n_samples)
    mean_imag = means.imag

    # --- mixture aggregation ---
    mixture_mean = means.mean(dim=0)  # (n_samples,)

    e_var_real = var_reals.mean(dim=0)  # aleatoric real
    e_var_imag = var_imags.mean(dim=0)  # aleatoric imag
    aleatoric_ri = cov.mean(dim=0)

    diff_r = mean_real - mean_real.mean(dim=0)
    diff_i = mean_imag - mean_imag.mean(dim=0)
    var_e_real = (diff_r ** 2).mean(dim=0)  # epistemic real
    var_e_imag = (diff_i ** 2).mean(dim=0)  # epistemic imag
    epistemic_ri = (diff_r * diff_i).mean(dim=0)

    mixture_var_real = (e_var_real + var_e_real).clamp(min=1e-6)
    mixture_var_imag = (e_var_imag + var_e_imag).clamp(min=1e-6)
    total_cov = mixture_var_real + mixture_var_imag

    # convert covariance → rho
    rho = total_cov / torch.sqrt(mixture_var_real * mixture_var_imag + 1e-6)

    aleatoric = (e_var_real + e_var_imag).mean().item()
    epistemic = (var_e_real + var_e_imag).mean().item()
    predictive = aleatoric + epistemic

    # --- RMSE ---
    rmse = torch.sqrt((torch.abs(mixture_mean - y) ** 2).mean()).item()

    # --- NLL ---
    loss_func = NegativeLogLossComplex()
    nll = loss_func(mixture_mean, (mixture_var_real, mixture_var_imag), rho, y).item()

    # --- R² ---
    ss_res = (torch.abs(mixture_mean - y) ** 2).sum()
    ss_tot = (torch.abs(y - y.mean()) ** 2).sum()
    r_squared = (1 - ss_res / ss_tot).item()

    # --- PICP ---
    # proportion of true values falling within the 95% prediction interval
    # for independent real/imag Gaussians: z_975 = 1.96
    z = 1.96
    # z = 3.29
    in_interval_real = (
            (y.real >= mixture_mean.real - z * mixture_var_real.sqrt()) &
            (y.real <= mixture_mean.real + z * mixture_var_real.sqrt())
    ).float().mean().item()

    in_interval_imag = (
            (y.imag >= mixture_mean.imag - z * mixture_var_imag.sqrt()) &
            (y.imag <= mixture_mean.imag + z * mixture_var_imag.sqrt())
    ).float().mean().item()

    picp = (in_interval_real + in_interval_imag) / 2
    # well calibrated model should be close to 0.95
    picp_error = abs(picp - 0.95)

    # --- MPIW ---
    # mean width of the prediction intervals across real and imaginary parts
    mpiw_real = (2 * z * mixture_var_real.sqrt()).mean().item()
    mpiw_imag = (2 * z * mixture_var_imag.sqrt()).mean().item()

    mpiw = (mpiw_real + mpiw_imag) / 2

    # --- calibration curve + ECE ---
    confidence_levels = torch.linspace(0.01, 0.99, 99)
    observed_coverage_real = []
    observed_coverage_imag = []

    for p in confidence_levels:
        z_p = torch.distributions.Normal(0, 1).icdf((1 + p) / 2)

        cov_real = (
                (y.real >= mixture_mean.real - z_p * mixture_var_real.sqrt()) &
                (y.real <= mixture_mean.real + z_p * mixture_var_real.sqrt())
        ).float().mean().item()

        cov_imag = (
                (y.imag >= mixture_mean.imag - z_p * mixture_var_imag.sqrt()) &
                (y.imag <= mixture_mean.imag + z_p * mixture_var_imag.sqrt())
        ).float().mean().item()

        observed_coverage_real.append(cov_real)
        observed_coverage_imag.append(cov_imag)

    observed_coverage_real = np.array(observed_coverage_real)
    observed_coverage_imag = np.array(observed_coverage_imag)
    expected_coverage = confidence_levels.numpy()

    # ECE — mean absolute deviation from the diagonal
    ece_real = np.abs(observed_coverage_real - expected_coverage).mean()
    ece_imag = np.abs(observed_coverage_imag - expected_coverage).mean()
    ece = (ece_real + ece_imag) / 2

    # --- calibration plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, observed, component, ece_val in zip(
            axes,
            [observed_coverage_real, observed_coverage_imag],
            ['Real', 'Imaginary'],
            [ece_real, ece_imag]
    ):
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(expected_coverage, observed, label=f'Model (ECE={ece_val:.3f})')
        ax.fill_between(expected_coverage, expected_coverage, observed,
                        alpha=0.2, color='red', label='Calibration gap')
        ax.set_xlabel('Expected coverage')
        ax.set_ylabel('Observed coverage')
        ax.set_title(f'Calibration curve — {component} part')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Residual vs Prediction Std Dev ---
    residuals = torch.abs(mixture_mean - y).cpu().numpy()
    pred_std = torch.sqrt(mixture_var_real + mixture_var_imag).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color gradient
    scatter = ax.scatter(pred_std, residuals, alpha=0.6, s=40, c=residuals,
                         cmap='plasma', edgecolors='black', linewidth=0.5)

    # Add trend line
    z = np.polyfit(pred_std, residuals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pred_std.min(), pred_std.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2.5, label=f'Trend (slope={z[0]:.3f})')

    # Calculate correlation
    corr = np.corrcoef(pred_std, residuals)[0, 1]

    ax.set_xlabel('Predicted Standard Deviation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Residual', fontsize=12, fontweight='bold')
    ax.set_title(f'Residual vs Prediction Uncertainty\n(Pearson r = {corr:.3f})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Residual Magnitude', fontsize=10)

    plt.tight_layout()
    plt.show()

    # --- VIZ 7: Coverage vs Confidence Level ---
    # Calculate observed coverage at multiple confidence levels
    confidence_levels_detailed = torch.linspace(0.50, 0.99, 50)
    observed_coverage_all = []

    for p in confidence_levels_detailed:
        z_p = torch.distributions.Normal(0, 1).icdf((1 + p) / 2)

        cov_real = (
                (y.real >= mixture_mean.real - z_p * mixture_var_real.sqrt()) &
                (y.real <= mixture_mean.real + z_p * mixture_var_real.sqrt())
        ).float().mean().item()

        cov_imag = (
                (y.imag >= mixture_mean.imag - z_p * mixture_var_imag.sqrt()) &
                (y.imag <= mixture_mean.imag + z_p * mixture_var_imag.sqrt())
        ).float().mean().item()

        observed_coverage_all.append((cov_real + cov_imag) / 2)

    observed_coverage_all = np.array(observed_coverage_all)
    confidence_levels_detailed = confidence_levels_detailed.numpy()

    fig, ax = plt.subplots(figsize=(11, 6))

    # Perfect calibration line
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=2.5, label='Perfect calibration', alpha=0.7)

    # Observed coverage
    ax.plot(confidence_levels_detailed, observed_coverage_all, 'o-', linewidth=2.5,
            markersize=6, label='Model coverage', color='#FF6B6B', markeredgecolor='darkred',
            markeredgewidth=1)

    # Highlight key confidence levels
    key_levels = [0.68, 0.90, 0.95, 0.99]
    for level in key_levels:
        idx = np.argmin(np.abs(confidence_levels_detailed - level))
        obs_cov = observed_coverage_all[idx]
        ax.plot(level, obs_cov, 'D', markersize=10, color='#4ECDC4',
                markeredgecolor='darkgreen', markeredgewidth=1.5)
        ax.annotate(f'{obs_cov:.3f}', xy=(level, obs_cov), xytext=(10, 10),
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax.fill_between(confidence_levels_detailed, confidence_levels_detailed,
                    observed_coverage_all, alpha=0.2, color='red', label='Calibration gap')

    ax.set_xlabel('Confidence Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Coverage', fontsize=12, fontweight='bold')
    ax.set_title('Coverage vs Confidence Level (Multi-Level PICP)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([0.4, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    plt.show()

    # --- print summary ---
    print(f"RMSE                  = {rmse:.4f}")
    print(f"NLL                   = {nll:.4f}")
    print(f"R²                    = {r_squared:.4f}")
    print(f"PICP (95%)            = {picp:.4f}  (ideal: 0.95, error: {picp_error:.4f})")
    print(f"MPIW                  = {mpiw:.4f}")
    print(f"ECE                   = {ece:.4f}  (lower is better)")
    print(f"Predictive uncertainty = {predictive:.4f}")
    print(f"  Aleatoric            = {aleatoric:.4f}")
    print(f"  Epistemic            = {epistemic:.4f}")

    return {
        "rmse": rmse,
        "nll": nll,
        "r_squared": r_squared,
        "picp": picp,
        "picp_error": picp_error,
        "ece": ece,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "predictive": predictive,
        "calibration": {
            "expected": expected_coverage,
            "observed_real": observed_coverage_real,
            "observed_imag": observed_coverage_imag,
        }
    }
