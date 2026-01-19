import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


from complexPytorch import CrossEntropyComplex, CrossEntropyComplexTwice
from deepensemble.ensemble import DeepEnsemble
from deepensemble.uncertainty import compute_and_plot_auroc
from heatmap import calibration_curve


def ensemble_experiment(real_network, complex_network,train_loader, val_loader, test_loader,output_dim,
                        num_models=5, device='cuda',epochs=10, ood=False, ood_loader = None):
    real_ensemble = DeepEnsemble(real_network, num_models, {'num_classes': output_dim}, device)
    real_ensemble.train_ensemble(train_loader,val_loader,epochs, loss_func=torch.nn.CrossEntropyLoss())

    complex_ensemble = DeepEnsemble(complex_network, num_models, {'num_classes': output_dim}, device)
    complex_ensemble.train_ensemble(train_loader, val_loader, epochs, loss_func=torch.nn.CrossEntropyLoss())

    print("Testing complex ensemble...")
    complex_metrics = complex_ensemble.test_ensemble(test_loader, torch.nn.CrossEntropyLoss())
    print("Testing real ensemble...")
    real_metrics = real_ensemble.test_ensemble(test_loader, torch.nn.CrossEntropyLoss())

    if ood:
        print("OOD Testing")
        complex_ood_metrics = complex_ensemble.test_ensemble(ood_loader, torch.nn.CrossEntropyLoss(), ood=True)
        real_ood_metrics = real_ensemble.test_ensemble(ood_loader, torch.nn.CrossEntropyLoss(), ood=True)
        print(f"Complex Network: OOD Confidence: {complex_ood_metrics['max_prob_ood']}")
        print(f"Complex Network: OOD MI: {complex_ood_metrics['mutual_information_ood']}")
        print(f"Real Network: OOD Confidence: {real_ood_metrics['max_prob_ood']}")
        print(f"Real Network: OOD MI: {real_ood_metrics['mutual_information_ood']}")

    print(f"Complex Network ID Confidence: {complex_metrics['max_prob_id']}")
    print(f"Complex Network ID MI: {complex_metrics['mutual_information_id']}")
    print(f"Real Network: ID Confidence: {real_metrics['max_prob_id']}")
    print(f"Real Network: ID MI: {real_metrics['mutual_information_id']}")

    if ood:
        categories = ['ID Data', 'OOD Data']
        set1_avgs = [real_metrics['mutual_information_id'],real_ood_metrics['mutual_information_ood']]
        set2_avgs = [complex_metrics['mutual_information_id'], complex_ood_metrics['mutual_information_ood']]
        x = np.arange(len(categories))
        width = 0.35

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bars
        bars1 = ax.bar(x - width / 2, set1_avgs, width, label="Real network", alpha=0.8)
        bars2 = ax.bar(x + width / 2, set2_avgs, width, label="Complex network", alpha=0.8)

        # Customize plot
        ax.set_xlabel('Data Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mutual Information', fontsize=12, fontweight='bold')
        ax.set_title('Ensemble Mutual Information on ID vs OOD Data',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.legend()
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 7))

        real_auroc = compute_and_plot_auroc(
            real_metrics['mi_scores'],
            real_ood_metrics['mi_scores'],
            label="Real Ensemble",
            ax=ax
        )

        complex_auroc = compute_and_plot_auroc(
            complex_metrics['mi_scores'],
            complex_ood_metrics['mi_scores'],
            label="Complex Ensemble",
            ax=ax
        )

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("OOD Detection AUROC (Mutual Information)")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.show()

        print(f"Real Ensemble MI AUROC: {real_auroc:.4f}")
        print(f"Complex Ensemble MI AUROC: {complex_auroc:.4f}")

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


