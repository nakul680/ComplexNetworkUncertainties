import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from BLL.bayesianlastlayer import BLLModel
from heatmap import fpr_at_95_tpr, calibration_curve, plot_calibration, count_params


def bll_experiment(real_network, complex_network, train_loader, valid_loader, test_loader, output_dim=11, epochs=10,
                   lr=0.005, ood=False, ood_dataloader=None, feature_dim=128):
    logs = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "nll": []

    }

    complex_logs = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "nll": []

    }

    complex_model = BLLModel(complex_network, output_dim, feature_dim, {'num_classes': output_dim}, 'cuda', train_loader)
    complex_params = count_params(complex_model)

    real_model = BLLModel(real_network, output_dim, feature_dim, {'num_classes': output_dim}, 'cuda', train_loader)
    real_params = count_params(real_model)
    print(f"Real Network Parameters: {real_params}")

    print(f"Complex Network Parameters: {complex_params}")
    complex_model.train_model(epochs, train_loader, valid_loader, torch.optim.Adam(complex_model.parameters(), lr=lr),
                              complex_logs)



    real_model.train_model(epochs, train_loader, valid_loader, torch.optim.Adam(real_model.parameters(), lr=lr), logs)

    print("Testing complex model\n")
    complex_preds, complex_entropy, complex_max_probs, complex_testlabels = complex_model.test_model(test_loader)
    print("Testing real model\n")
    real_preds, real_entropy, real_max_probs, real_testlabels = real_model.test_model(test_loader)

    if ood:
        print("Testing complex model on ood data\n")
        _, complex_entropy_ood, complex_ood_max_probs = complex_model.test_model(ood_dataloader, is_ood=True)
        print("Testing real model on ood data\n")
        _, real_entropy_ood, real_ood_max_probs = real_model.test_model(ood_dataloader, is_ood=True)

        complex_id_scores = complex_entropy.numpy()
        complex_ood_scores = complex_entropy_ood.numpy()

        real_id_scores = real_entropy.numpy()
        real_ood_scores = real_entropy_ood.numpy()

        complex_scores = np.concatenate((complex_id_scores,complex_ood_scores), axis=0)
        real_scores = np.concatenate((real_id_scores, real_ood_scores), axis=0)
        complex_labels = np.concatenate([np.zeros(len(complex_id_scores)),np.ones(len(complex_ood_scores))])
        real_labels = np.concatenate([np.zeros(len(real_id_scores)),np.ones(len(real_ood_scores))])

        complex_auroc = roc_auc_score(complex_labels, complex_scores)
        print("Complex Model AUROC: {:.4f}".format(complex_auroc))
        real_auroc = roc_auc_score(real_labels, real_scores)
        print("Real Model AUROC: {:.4f}".format(real_auroc))

        complex_fpr,complex_tpr,_ = roc_curve(complex_labels, complex_scores)
        print(f"Complex Model FPR@95TPR: {fpr_at_95_tpr(complex_fpr, complex_tpr)}")
        real_fpr,real_tpr,_ = roc_curve(real_labels, real_scores)
        print(f"Real Model FPR@95TPR: {fpr_at_95_tpr(real_fpr, real_tpr)}")

        plt.figure()
        plt.plot(real_fpr, real_tpr, label=f"Real Network AUROC = {real_auroc:.3f}")
        plt.plot(complex_fpr, complex_tpr, label=f"Complex Network AUROC = {complex_auroc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate (ID → OOD)")
        plt.ylabel("True Positive Rate (OOD detected)")
        plt.title("OOD Detection ROC Curve:Complex Model vs Real Model")
        plt.legend()
        plt.grid(True)
        plt.show()

    c_conf, c_acc, c_count = calibration_curve(
        complex_max_probs.numpy(),
        complex_preds.numpy(),
        complex_testlabels.numpy()
    )

    # Real model calibration
    r_conf, r_acc, r_count = calibration_curve(
        real_max_probs.numpy(),
        real_preds.numpy(),
        real_testlabels.numpy()
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




