import numpy as np
import torch

from deepensemble.uncertainty import calculate_classification_uncertainty


def test_models(models, device, test_dataloader, loss_func):
    """
    Test ensemble of models and calculate comprehensive uncertainty metrics.

    Args:
        models: List of model instances
        device: torch device
        test_dataloader: DataLoader for test data
        loss_func: Loss function (e.g., CrossEntropyLoss)

    Returns:
        dict with all metrics and predictions
    """
    # Set all models to evaluation mode
    for model in models:
        model.eval()

    num_models = len(models)
    all_ensemble_predictions = []  # Will store predictions for all batches
    all_labels = []  # Will store all true labels
    test_loss = 0.0

    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.float().to(device)
            y = y.long().to(device)

            # Collect predictions from all models for this batch
            batch_predictions = []
            for model in models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
                batch_predictions.append(probs.cpu())

            # Stack: (num_models, batch_size, num_classes)
            batch_predictions = torch.stack(batch_predictions, dim=0)

            # Calculate mean prediction for loss
            mean_pred = batch_predictions.mean(dim=0).to(device)
            loss = loss_func(mean_pred, y)
            test_loss += loss.item()

            # Store for later uncertainty calculation
            all_ensemble_predictions.append(batch_predictions)
            all_labels.append(y.cpu())

    # Concatenate all batches
    # Shape: (num_models, total_samples, num_classes)
    ensemble_predictions = torch.cat(all_ensemble_predictions, dim=1).numpy()
    true_labels = torch.cat(all_labels, dim=0).numpy()

    # Calculate comprehensive uncertainty metrics
    metrics = calculate_classification_uncertainty(ensemble_predictions, true_labels)

    # Add loss to metrics
    metrics['test_loss'] = test_loss / len(test_dataloader)

    # Calculate additional uncertainty metrics
    epsilon = 1e-10

    # Mutual Information
    mean_predictions = ensemble_predictions.mean(axis=0)
    pred_entropy = -np.sum(mean_predictions * np.log(mean_predictions + epsilon), axis=1)
    individual_entropy = -np.sum(
        ensemble_predictions * np.log(ensemble_predictions + epsilon),
        axis=2
    )
    exp_entropy = individual_entropy.mean(axis=0)
    mutual_info = pred_entropy - exp_entropy

    metrics['mutual_information'] = mutual_info.mean()
    metrics['mi_per_sample'] = mutual_info
    metrics['expected_entropy'] = exp_entropy.mean()

    # Prediction variance (epistemic uncertainty)
    prediction_variance = ensemble_predictions.var(axis=0)
    metrics['avg_prediction_variance'] = prediction_variance.mean()
    metrics['variance_per_sample'] = prediction_variance.mean(axis=1)

    # Print summary
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test NLL: {metrics['nll']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"Avg Predictive Entropy: {metrics['predictive_entropy']:.4f}")
    print(f"Avg Mutual Information: {metrics['mutual_information']:.4f}")
    print(f"Avg Prediction Variance: {metrics['avg_prediction_variance']:.4f}")

    return metrics
