import matplotlib.pyplot as plt
import numpy as np

from heatmap import plot_calibration


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


def plot_calibrations(confs, accs, names, bin_edges):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    widths = np.diff(bin_edges)
    bin_centers = [5,15,25,35,45,55,65,75,85,95]

    for ax, acc, name in zip(axes, accs, names):
        ax.bar(bin_centers, height=acc, width=widths, edgecolor='black', color='steelblue', alpha=0.3, label='Bins')
        ax.plot(confs, acc, marker='o')
        ax.plot([0, 100], [0, 100], '--', color='red', label='Perfect calibration')
        ax.set_xlim(0,100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Predictive confidence')
        ax.set_ylabel('Empirical accuracy')
        ax.set_title(f'{name}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # compare_histograms()
    plot_calibrations([5,15,25,35,45,55,65,75,85,95],[[5,15,25,35,45,55,65,75,85,95],[0,5,15,25,35,45,55,65,75,85],[15,25,35,45,55,65,75,85,95,100]],
                      names =['Perfectly calibrated', 'Overconfident', 'Underconfident'],
                      bin_edges=[0,10,20,30,40,50,60,70,80,90,100])