import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

from gender_age_bias_tests import get_biases_test_distribution

plt.rcParams.update(figsizes.icml2024_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})
plt.rcParams.update(bundles.icml2024(column='half'))


def main(results_age_df, results_gender_df):

    fig, ax = plt.subplots()

    rng = np.random.default_rng(42)
    pvalue_age = results_age_df['p_value'].to_numpy()
    pvalue_gender = results_gender_df['p_value'].to_numpy()

    y_age = rng.normal(loc=0.5, scale=0.1, size=pvalue_age.shape)
    y_gender = rng.normal(loc=-0.5, scale=0.1, size=pvalue_gender.shape)
    ax.scatter(pvalue_age, y_age, c=palettes.tue_plot[0], s=10, alpha=0.6, edgecolors='none')
    ax.axvline(x=0.05, color=palettes.tue_plot[1], linestyle='--', linewidth=1, label=f'p-value: 0.05')
    ax.scatter(pvalue_gender, y_gender, c=palettes.tue_plot[1],  s=10, alpha=0.6, edgecolors='none')
    ax.grid()
    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=palettes.tue_plot[0], label='Age'),
        Patch(facecolor=palettes.tue_plot[1], label='Gender'),
        Line2D([0], [0], color=palettes.tue_plot[1], linestyle='--', linewidth=1, label='p-value: 0.05')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel('p-value')
    plt.savefig('bias_test_distribution.pdf')

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, '..', '..', 'data', 'newspaper_collection_evaluation_results_20_12_2025.csv')
    age, gender = get_biases_test_distribution(path)
    main(age, gender)

