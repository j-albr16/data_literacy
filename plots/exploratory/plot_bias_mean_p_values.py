import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

from gender_age_bias_tests import get_biases_test_distribution

plt.rcParams.update(figsizes.icml2024_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})
plt.rcParams.update(bundles.icml2024(column='half'))

def main(path, results_age_df, results_gender_df):

    df = pd.read_csv(path)
    newspapers = df['newspaper'].unique()
    targets = ['Gender', 'Age']
    
    # Custom label map for newspaper names
    newspaper_label_map = {
        'sz': 'SZ',
        'spiegel': 'Spiegel',
        'stern': 'Stern',
        'taz': 'Taz',
        'compact': 'Compact',
        'freitag': 'Freitag',
        'nd': 'ND'
    }

    plt.savefig('dist_bias_tests.pdf')

    fig, ax = plt.subplots()

    results = {'Age': [], 'Gender': []}
    for j, n in enumerate(newspapers):
        df_ = results_gender_df
        df_ = df_[df_['newspaper'] == n]
        results['Gender'].append(df_['p_value'].mean())

        df_ = results_age_df
        df_ = df_[df_['newspaper'] == n]
        results['Age'].append(df_['p_value'].mean())

    x = np.arange(len(newspapers)) 
    width = 0.25
    multiplier = 0

    for attribute, measurement in results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, fmt='%.2f', padding=3, rotation=90, clip_on=True)
        multiplier += 1

    ax.set_ylabel('Mean p-value')
    ax.set_xticks(x + width, [newspaper_label_map.get(n, n.capitalize()) for n in newspapers])
    ax.legend(loc='upper right', ncols=2)
    ax.set_ylim(0, 1)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
    ax.grid(True, alpha=0.3)

    plt.savefig('bias_mean_p_values.pdf')


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    path = os.path.join(script_dir, '..', '..', 'data', 'newspaper_collection_evaluation_results_20_12_2025.csv')
    age, gender = get_biases_test_distribution(path)
    main(path, age, gender)




