import matplotlib.pyplot as plt
import numpy as np


def compare_histograms():
    """
    Plot two histograms on the same scale for comparison.
    """
    categories = ['Confidence on ID', 'Confidence on OOD']
    set1_avgs = [0.515,0.444]
    set2_avgs = [0.496,0.416]
    set3_avgs = [0.475, 0.396] #real
    set4_avgs = [0.486, 0.411]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - 1.5 * width, set1_avgs, width, label='Real Network', alpha=0.8)
    ax.bar(x - 0.5 * width, set2_avgs, width, label='Complex Network', alpha=0.8)
    ax.bar(x + 0.5 * width, set3_avgs, width, label='Real Ensemble', alpha=0.8)
    ax.bar(x + 1.5 * width, set4_avgs, width, label='Complex Ensemble', alpha=0.8)

    # Labels and formatting
    ax.set_xlabel('Prediction Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Max Probability', fontsize=12, fontweight='bold')
    ax.set_title(
        'Comparison of Average Max Confidence\nfor ID vs OOD Samples',
        fontsize=14,
        fontweight='bold'
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_histograms()