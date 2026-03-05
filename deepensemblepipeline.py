import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


from complexPytorch import CrossEntropyComplex, CrossEntropyComplexTwice
from complexnetwork.complexCNN import ComplexCNN_RLL
from deepensemble.ensemble import DeepEnsemble
from deepensemble.temp_scaling import TemperatureScaling
from deepensemble.uncertainty import compute_and_plot_auroc
from heatmap import calibration_curve, count_params, accuracy_rejection_curve
from realnetwork.amc_cnn import AMC_CNN




def ensemble_experiment(real_network, complex_network,train_loader, val_loader, test_loader,output_dim,
                        num_models=5, device='cuda',epochs=10, ood=False, ood_loader = None):
    real_temp_scaler = TemperatureScaling('cuda')
    complex_temp_scaler = TemperatureScaling('cuda')
    real_ensemble = DeepEnsemble(real_network, num_models, {'num_classes': output_dim}, device=device)
    real_ensemble.train_ensemble(train_loader,val_loader,epochs, loss_func=torch.nn.CrossEntropyLoss())
    print(f"real network:{count_params(AMC_CNN(num_classes=output_dim))}")
    print(f"complex network:{count_params(ComplexCNN_RLL(num_classes=output_dim))}")
    complex_ensemble = DeepEnsemble(complex_network, num_models, {'num_classes': output_dim}, device=device)
    complex_ensemble.train_ensemble(train_loader, val_loader, epochs, loss_func=torch.nn.CrossEntropyLoss())


    # complex_temp_scaler.fit(complex_ensemble, val_loader)
    print("Testing complex ensemble...")
    complex_metrics = complex_ensemble.test_ensemble(test_loader)
    # real_temp_scaler.fit(real_ensemble, val_loader)
    print("Testing real ensemble...")
    real_metrics = real_ensemble.test_ensemble(test_loader)

    if ood:
        print("OOD Testing")
        complex_ood_metrics = complex_ensemble.test_ensemble(ood_loader)
        real_ood_metrics = real_ensemble.test_ensemble(ood_loader)
        print(f"Complex Network: OOD Confidence: {complex_ood_metrics['max_prob_ood']}")
        print(f"Complex Network: OOD MI: {complex_ood_metrics['mutual_information_ood']}")
        print(f"Complex Network: OOD Entropy: {complex_ood_metrics['predictive_entropy_ood']}")
        print(f"Real Network: OOD Confidence: {real_ood_metrics['max_prob_ood']}")
        print(f"Real Network: OOD MI: {real_ood_metrics['mutual_information_ood']}")
        print(f"Real Network: OOD Entropy: {real_ood_metrics['predictive_entropy_ood']}")

    print(f"Complex Network ID Confidence: {complex_metrics['max_prob_id']}")
    print(f"Complex Network ID MI: {complex_metrics['mutual_information_id']}")
    print(F"Complex Network ID Entropy: {complex_metrics['predictive_entropy_id']}")
    print(f"Real Network: ID Confidence: {real_metrics['max_prob_id']}")
    print(f"Real Network: ID MI: {real_metrics['mutual_information_id']}")
    print(f"Real Network: ID Entropy: {real_metrics['predictive_entropy_id']}")

    if ood:
        categories = ['ID Data', 'OOD Data']
        set1_avgs = [real_metrics['predictive_entropy_id'],real_ood_metrics['predictive_entropy_ood']]
        set2_avgs = [complex_metrics['predictive_entropy_id'], complex_ood_metrics['predictive_entropy_ood']]
        x = np.arange(len(categories))
        width = 0.35

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bars
        bars1 = ax.bar(x - width / 2, set1_avgs, width, label="Real network", alpha=0.8)
        bars2 = ax.bar(x + width / 2, set2_avgs, width, label="Complex network", alpha=0.8)

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
            label="Real Ensemble",
            ax=ax
        )

        complex_auroc = compute_and_plot_auroc(
            complex_metrics['all_entropy'],
            complex_ood_metrics['all_entropy'],
            label="Complex Ensemble",
            ax=ax
        )

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("OOD Detection AUROC (Predictive Entropy)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.show()

        print(f"Real Ensemble AUROC: {real_auroc:.4f}")
        print(f"Complex Ensemble AUROC: {complex_auroc:.4f}")

    c_conf, c_acc, c_count = calibration_curve(
        complex_metrics['max_probs'].numpy(),
        complex_metrics['predictions'].numpy(),
        complex_metrics['labels'].numpy()
    )

    # Real model calibration
    r_conf, r_acc, r_count = calibration_curve(
        real_metrics['max_probs'].numpy(),
        real_metrics['predictions'].numpy(),
        real_metrics['labels'].numpy()
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

    real_rej, real_acc = accuracy_rejection_curve(real_metrics['labels'].detach().numpy(), real_metrics['predictions'].detach().numpy(), real_metrics['max_probs'].detach().numpy())
    complex_rej, complex_acc = accuracy_rejection_curve(complex_metrics['labels'].detach().numpy(), complex_metrics['predictions'].detach().numpy(), complex_metrics['max_probs'].detach().numpy())


    plt.plot(real_rej, real_acc, label=" Real Accuracy-Rejection Curve")
    plt.plot(complex_rej, complex_acc, label=" Complex Accuracy-Rejection Curve")
    plt.xlabel("Rejection Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy-Rejection Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

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

    ax[0].set_title("Real Network")
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

    ax[1].set_title("Complex Network")
    ax[1].set_xlabel("Predicted probability")

    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()
