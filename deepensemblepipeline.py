import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from complexPytorch import CrossEntropyComplex, CrossEntropyComplexTwice
from complexPytorch.complexLayers import RealGaussianNLL
from complexPytorch.complexLoss import NegativeLogLossComplex
from complexnetwork.complexCNN import ComplexCNN_RLL, ComplexCNN_ABS
from deepensemble.ensemble import DeepEnsemble
from deepensemble.temp_scaling import TemperatureScaling
from deepensemble.uncertainty import compute_and_plot_auroc
from heatmap import calibration_curve, count_params, accuracy_rejection_curve
from realnetwork.amc_cnn import AMC_CNN


def ensemble_experiment(real_network, complex_network, train_loader, val_loader, test_loader, output_dim=None, lag=None,
                        num_models=5, device='cuda', epochs=10, ood=False, ood_loader=None, task='classification', seed=None):
    real_temp_scaler = TemperatureScaling('cuda')
    complex_temp_scaler = TemperatureScaling('cuda')
    if task == 'classification':
        real_ensemble = DeepEnsemble(real_network, num_models, {'num_classes': output_dim}, device=device, seed = seed)
        real_ensemble.train_ensemble(train_loader, val_loader, epochs, loss_func=torch.nn.CrossEntropyLoss())
        print(f"real network:{count_params(AMC_CNN(num_classes=output_dim))}")
        print(f"complex network:{count_params(ComplexCNN_RLL(num_classes=output_dim))}")
        complex_ensemble = DeepEnsemble(complex_network, num_models, {'num_classes': output_dim}, device=device, seed = seed)
        complex_ensemble.train_ensemble(train_loader, val_loader, epochs, loss_func=torch.nn.CrossEntropyLoss())
        complex_ensemble1 = DeepEnsemble(ComplexCNN_ABS, num_models, {'num_classes': output_dim}, device=device, seed = seed)
        complex_ensemble1.train_ensemble(train_loader, val_loader, epochs, loss_func=torch.nn.CrossEntropyLoss())
    else:
        real_ensemble = DeepEnsemble(real_network, num_models, {'lag': lag}, task='regression', device=device)
        real_ensemble.train_ensemble(train_loader, val_loader, epochs, loss_func=NegativeLogLossComplex())
        complex_ensemble = DeepEnsemble(complex_network, num_models, {'lag': lag}, task='regression', device=device)
        complex_ensemble.train_ensemble(train_loader, val_loader, epochs, loss_func=NegativeLogLossComplex())

    # complex_temp_scaler.fit(complex_ensemble, val_loader)
    print(f"Testing {complex_ensemble.backbone.name}")
    complex_metrics = complex_ensemble.test_ensemble(test_loader)
    # real_temp_scaler.fit(real_ensemble, val_loader)
    print(f"Testing {real_ensemble.backbone.name}")
    real_metrics = real_ensemble.test_ensemble(test_loader)
    print(f"Testing {complex_ensemble1.backbone.name}")
    complex_metrics1 = complex_ensemble1.test_ensemble(test_loader)

    if task == 'classification':
        print(f"{complex_ensemble.backbone.name} ID Confidence: {complex_metrics['max_prob_id']}")
        print(f"{complex_ensemble.backbone.name} ID MI: {complex_metrics['mutual_information_id']}")
        print(f"{complex_ensemble.backbone.name} ID Entropy: {complex_metrics['predictive_entropy_id']}")
        print(f"{real_ensemble.backbone.name} ID Confidence: {real_metrics['max_prob_id']}")
        print(f"{real_ensemble.backbone.name} ID MI: {real_metrics['mutual_information_id']}")
        print(f"{real_ensemble.backbone.name} ID Entropy: {real_metrics['predictive_entropy_id']}")
        print(f"{complex_ensemble1.backbone.name} ID Confidence: {complex_metrics1['max_prob_id']}")
        print(f"{complex_ensemble1.backbone.name} ID MI: {complex_metrics1['mutual_information_id']}")
        print(f"{complex_ensemble1.backbone.name} ID Entropy: {complex_metrics1['predictive_entropy_id']}")

    if ood:
        print("OOD Testing")
        complex_ood_metrics = complex_ensemble.test_ensemble(ood_loader)
        real_ood_metrics = real_ensemble.test_ensemble(ood_loader)
        complex_ood_metrics1 = complex_ensemble1.test_ensemble(ood_loader)
        if task == 'classification':
            print(f"{complex_ensemble.backbone.name} OOD Confidence: {complex_ood_metrics['max_prob_ood']}")
            print(f"{complex_ensemble.backbone.name} OOD MI: {complex_ood_metrics['mutual_information_ood']}")
            print(f"{complex_ensemble.backbone.name} OOD Entropy: {complex_ood_metrics['predictive_entropy_ood']}")
            print(f"{real_ensemble.backbone.name} OOD Confidence: {real_ood_metrics['max_prob_ood']}")
            print(f"{real_ensemble.backbone.name} OOD MI: {real_ood_metrics['mutual_information_ood']}")
            print(f"{real_ensemble.backbone.name} OOD Entropy: {real_ood_metrics['predictive_entropy_ood']}")
            print(f"{complex_ensemble1.backbone.name} OOD Confidence: {complex_ood_metrics1['max_prob_ood']}")
            print(f"{complex_ensemble1.backbone.name} OOD MI: {complex_ood_metrics1['mutual_information_ood']}")
            print(f"{complex_ensemble1.backbone.name} OOD Entropy: {complex_ood_metrics1['predictive_entropy_ood']}")

            categories = ['ID Data', 'OOD Data']
            set1_avgs = [real_metrics['predictive_entropy_id'], real_ood_metrics['predictive_entropy_ood']]
            set2_avgs = [complex_metrics['predictive_entropy_id'], complex_ood_metrics['predictive_entropy_ood']]
            set3_avgs = [complex_metrics1['predictive_entropy_id'], complex_ood_metrics1['predictive_entropy_ood']]
            x = np.arange(len(categories))
            width = 0.2

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create bars
            bars1 = ax.bar(x - width, set1_avgs, width, label=f"{real_ensemble.backbone.name}", alpha=0.8)
            bars2 = ax.bar(x, set2_avgs, width, label=f"{complex_ensemble.backbone.name}", alpha=0.8)
            bars3 = ax.bar(x + width, set3_avgs, width, label=f"{complex_ensemble1.backbone.name}", alpha=0.8)
            # Customize plot
            ax.set_xlabel('Data Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predictive Entropy', fontsize=12, fontweight='bold')
            ax.set_title('Ensemble Predictive Entropy on ID vs OOD Data',
                         fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.legend()
            plt.show()

            fig, ax = plt.subplots(figsize=(7, 7))

            real_auroc = compute_and_plot_auroc(
                real_metrics['all_entropy'],
                real_ood_metrics['all_entropy'],
                label=f"{real_ensemble.backbone.name} Ensemble",
                ax=ax
            )

            complex_auroc = compute_and_plot_auroc(
                complex_metrics['all_entropy'],
                complex_ood_metrics['all_entropy'],
                label=f"{complex_ensemble.backbone.name}",
                ax=ax
            )

            complex_auroc1 = compute_and_plot_auroc(
                complex_metrics1['all_entropy'],
                complex_ood_metrics1['all_entropy'],
                label=f"{complex_ensemble1.backbone.name}",
                ax=ax
            )

            ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("OOD Detection AUROC (Predictive Entropy)")
            ax.legend()
            ax.grid(alpha=0.3)

            plt.show()

            print(f"{real_ensemble.backbone.name} Ensemble AUROC: {real_auroc:.4f}")
            print(f"{complex_ensemble.backbone.name} Ensemble AUROC: {complex_auroc:.4f}")
            print(f"{complex_ensemble1.backbone.name} Ensemble AUROC: {complex_auroc1:.4f}")

    c_conf, c_acc, c_count = calibration_curve(
        complex_metrics['max_probs'].numpy(),
        complex_metrics['predictions'].numpy(),
        complex_metrics['labels'].numpy()
    )

    c1_conf, c1_acc, c1_count = calibration_curve(
        complex_metrics1['max_probs'].numpy(),
        complex_metrics1['predictions'].numpy(),
        complex_metrics1['labels'].numpy()
    )

    # Real model calibration
    r_conf, r_acc, r_count = calibration_curve(
        real_metrics['max_probs'].numpy(),
        real_metrics['predictions'].numpy(),
        real_metrics['labels'].numpy()
    )

    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.plot(c_conf, c_acc, marker='o', label=f'{complex_ensemble.backbone.name} Ensemble')
    plt.plot(c1_conf, c1_acc, marker='o', label=f'{complex_ensemble1.backbone.name} Ensemble')
    plt.plot(r_conf, r_acc, marker='o', label=f'{real_ensemble.backbone.name} Ensemble')
    plt.legend()
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Comparison')
    plt.grid(True)
    plt.show()

    real_rej, real_acc = accuracy_rejection_curve(real_metrics['labels'].detach().numpy(),
                                                  real_metrics['predictions'].detach().numpy(),
                                                  real_metrics['max_probs'].detach().numpy())
    complex_rej, complex_acc = accuracy_rejection_curve(complex_metrics['labels'].detach().numpy(),
                                                        complex_metrics['predictions'].detach().numpy(),
                                                        complex_metrics['max_probs'].detach().numpy())

    complex_rej1, complex_acc1 = accuracy_rejection_curve(complex_metrics1['labels'].detach().numpy(),
                                                          complex_metrics1['predictions'].detach().numpy(),
                                                          complex_metrics1['max_probs'].detach().numpy())

    plt.plot(real_rej, real_acc, label=f"{real_ensemble.backbone.name} Ensemble Accuracy-Rejection Curve")
    plt.plot(complex_rej, complex_acc, label=f"{complex_ensemble.backbone.name} Ensemble Accuracy-Rejection Curve")
    plt.plot(complex_rej1, complex_acc1, label=f"{complex_ensemble1.backbone.name} Ensemble Accuracy-Rejection Curve")
    plt.xlabel("Rejection Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy-Rejection Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(10, 6), sharey=True)

    # Common x-range for probabilities
    x = np.linspace(0, 1, 500)

    # ---- Real network ----
    ax[0].hist(
        real_metrics['correct_max_probs'],
        bins='auto',
        density=True,
        alpha=0.5,
        label='Correct'
    )
    ax[0].hist(
        real_metrics['wrong_max_probs'],
        bins='auto',
        density=True,
        alpha=0.5,
        label='Wrong'
    )

    ax[0].plot(
        x,
        gaussian_kde(real_metrics['correct_max_probs'])(x)
    )
    ax[0].plot(
        x,
        gaussian_kde(real_metrics['wrong_max_probs'])(x)
    )
    real_jsd = distribution_divergence(real_metrics['correct_max_probs'],real_metrics['wrong_max_probs'], x)

    ax[0].set_title(f"{real_ensemble.backbone.name} Ensemble (JSD={real_jsd:.4f})")
    ax[0].set_xlabel("Predicted probability")
    ax[0].set_ylabel("Density")

    # ---- Complex network ----
    ax[1].hist(
        complex_metrics['correct_max_probs'],
        bins='auto',
        density=True,
        alpha=0.5,
        label='Correct'
    )
    ax[1].hist(
        complex_metrics['wrong_max_probs'],
        bins='auto',
        density=True,
        alpha=0.5,
        label='Wrong'
    )

    ax[1].plot(
        x,
        gaussian_kde(complex_metrics['correct_max_probs'])(x)
    )
    ax[1].plot(
        x,
        gaussian_kde(complex_metrics['wrong_max_probs'])(x)
    )
    complex_jsd = distribution_divergence(complex_metrics['correct_max_probs'], complex_metrics['wrong_max_probs'], x)


    ax[1].set_title(f"{complex_ensemble.backbone.name} Ensemble (JSD={complex_jsd:.4f})")
    ax[1].set_xlabel("Predicted probability")

    ax[2].hist(
        complex_metrics1['correct_max_probs'],
        bins='auto',
        density=True,
        alpha=0.5,
        label='Correct'
    )
    ax[2].hist(
        complex_metrics1['wrong_max_probs'],
        bins='auto',
        density=True,
        alpha=0.5,
        label='Wrong'
    )

    ax[2].plot(
        x,
        gaussian_kde(complex_metrics1['correct_max_probs'])(x)
    )
    ax[2].plot(
        x,
        gaussian_kde(complex_metrics1['wrong_max_probs'])(x)
    )
    complex1_jsd = distribution_divergence(complex_metrics1['correct_max_probs'], complex_metrics1['wrong_max_probs'], x)

    ax[2].set_title(f"{complex_ensemble1.backbone.name} Ensemble (JSD={complex1_jsd:.4f})")
    ax[2].set_xlabel("Predicted probability")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.tight_layout()
    plt.show()


def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric, always finite)."""
    m = 0.5 * (p + q)
    # Clip to avoid log(0)
    p, q, m = np.clip(p, 1e-10, None), np.clip(q, 1e-10, None), np.clip(m, 1e-10, None)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

def distribution_divergence(a, b, x):
    p = gaussian_kde(a)(x)
    q = gaussian_kde(b)(x)
    p /= p.sum(); q /= q.sum()  # normalize to sum to 1
    return js_divergence(p, q)