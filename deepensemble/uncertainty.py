import numpy as np
import matplotlib.pyplot as plt

def calculate_classification_uncertainty(ensemble_predictions, true_labels, batch_size=32):
    """
    Calculate uncertainty metrics for classification, with batching and batch-wise value plots.

    Args:
        ensemble_predictions: np.ndarray, shape (num_models, sample_size, num_classes)
        true_labels: np.ndarray, shape (sample_size,) - integer class labels
        batch_size: int, number of samples per batch

    Returns:
        dict with:
            - overall metrics (nll, accuracy, brier_score, predictive_entropy)
            - per-sample values
            - per-batch NLL and Brier score arrays
    """
    epsilon = 1e-10
    num_classes = ensemble_predictions.shape[2]
    sample_size = ensemble_predictions.shape[1]

    # 1. Average predictions across ensemble (predictive distribution)
    mean_predictions = ensemble_predictions.mean(axis=0)  # (sample_size, num_classes)

    # 2. Negative Log-Likelihood (NLL)
    true_class_probs = mean_predictions[np.arange(sample_size), true_labels]
    nll = -np.log(true_class_probs + epsilon)
    avg_nll = nll.mean()

    # 3. Classification Accuracy
    predicted_classes = mean_predictions.argmax(axis=1)
    accuracy = (predicted_classes == true_labels).mean()

    # 4. Brier Score
    t_star = np.zeros((sample_size, num_classes))
    t_star[np.arange(sample_size), true_labels] = 1
    brier_per_sample = np.sum((t_star - mean_predictions) ** 2, axis=1) / num_classes
    brier_score = brier_per_sample.mean()

    # 5. Predictive Entropy
    predictive_entropy = -np.sum(mean_predictions * np.log(mean_predictions + epsilon), axis=1)
    avg_entropy = predictive_entropy.mean()

    # ---- Predictive entropy for correct vs incorrect predictions ----
    correct_mask = (predicted_classes == true_labels)
    incorrect_mask = ~correct_mask

    entropy_correct = predictive_entropy[correct_mask]
    entropy_incorrect = predictive_entropy[incorrect_mask]

    avg_entropy_correct = entropy_correct.mean() if entropy_correct.size > 0 else np.nan
    avg_entropy_incorrect = entropy_incorrect.mean() if entropy_incorrect.size > 0 else np.nan


    # ---- Batching section ----
    num_batches = int(np.ceil(sample_size / batch_size))
    nll_batches = []
    brier_batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, sample_size)
        nll_batches.append(nll[start:end].mean())
        brier_batches.append(brier_per_sample[start:end].mean())

    nll_batches = np.array(nll_batches)
    brier_batches = np.array(brier_batches)

    # ---- Plot actual per-batch values ----
    # plt.figure(figsize=(12, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.hist(nll_batches, bins='auto', color='skyblue', edgecolor='black')
    # plt.title('Distribution of Average NLL per Batch')
    # plt.xlabel('Average NLL (per batch)')
    # plt.ylabel('Number of Batches')
    #
    # plt.subplot(1, 2, 2)
    # plt.hist(brier_batches, bins='auto', color='salmon', edgecolor='black')
    # plt.title('Distribution of Average Brier Score per Batch')
    # plt.xlabel('Average Brier Score (per batch)')
    # plt.ylabel('Number of Batches')
    #
    # plt.tight_layout()
    # plt.show()

    # ---- Plot predictive entropy: correct vs incorrect ----
    # plt.figure(figsize=(12, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.hist(entropy_correct, bins='auto', edgecolor='black')
    # plt.title('Predictive Entropy (Correct Predictions)')
    # plt.xlabel('Predictive Entropy')
    # plt.ylabel('Count')
    #
    # plt.subplot(1, 2, 2)
    # plt.hist(entropy_incorrect, bins='auto', edgecolor='black')
    # plt.title('Predictive Entropy (Incorrect Predictions)')
    # plt.xlabel('Predictive Entropy')
    # plt.ylabel('Count')
    #
    # plt.tight_layout()
    # plt.show()

    # categories = ['Uncertainty on Correct Predictions', 'Uncertainty on Wrong Predictions']
    # set1_avgs = [, 1 - sum(real_wrong) / len(real_wrong)]
    # set2_avgs = [1 - sum(complex_correct) / len(complex_correct), 1 - sum(complex_wrong) / len(complex_wrong)]
    # x = np.arange(len(categories))
    # width = 0.35
    #
    # # Create figure and axis
    # fig, ax = plt.subplots(figsize=(10, 6))
    #
    # # Create bars
    # bars1 = ax.bar(x - width / 2, set1_avgs, width, label="Real network", alpha=0.8)
    # bars2 = ax.bar(x + width / 2, set2_avgs, width, label="Complex network", alpha=0.8)
    #
    # # Customize plot
    # ax.set_xlabel('Prediction Type', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Average Max Distance', fontsize=12, fontweight='bold')
    # ax.set_title('Comparison of Average Max Distances\nfor Correct vs Wrong Predictions',
    #              fontsize=14, fontweight='bold')
    # ax.set_xticks(x)
    # ax.set_xticklabels(categories)
    # ax.legend(fontsize=11)
    # ax.grid(axis='y', alpha=0.3, linestyle='--')
    #
    # plt.legend()
    # plt.show()


    return {
        'nll': avg_nll,
        'nll_per_sample': nll,
        'accuracy': accuracy,
        'brier_score': brier_score,
        'brier_per_sample': brier_per_sample,
        'predictive_entropy': avg_entropy,
        'entropy_per_sample': predictive_entropy,
        'nll_per_batch': nll_batches,
        'brier_per_batch': brier_batches,
        'avg_correct_entropy': avg_entropy_correct,
        'avg_wrong_entropy': avg_entropy_incorrect
    }
