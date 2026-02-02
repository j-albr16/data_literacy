import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

plt.rcParams.update(figsizes.icml2024_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})
plt.rcParams.update(bundles.icml2024(column='half'))


def main(path, emotion='happy'):

    df = pd.read_csv(path)
    # make the confidence plot
    df_ = df[df['newspaper'] == 'compact'][[emotion, 'gender']]
    X = df_['happy'].to_numpy()
    is_male = (df_['gender'] == 'male').to_numpy()
    rng_scatter = np.random.default_rng(42)
    y = rng_scatter.normal(loc=0.5, scale=0.1, size=X.shape)
    cdf_x = np.linspace(0, 1, 1000)
    males = X[is_male]
    females = X[~is_male]
    male_cdf = np.array([len(males[males < x]) / len(males) for x in cdf_x])
    female_cdf = np.array([len(females[females < x]) / len(females) for x in cdf_x])
    epsilon_male = np.sqrt((1/(2*len(males))) * np.log(2 / 0.05) )
    epsilon_female = np.sqrt((1/(2*len(females))) * np.log(2 / 0.05) )
    colors = [palettes.tue_plot[0] if male else palettes.tue_plot[3] for male in is_male]

    fig, ax = plt.subplots()

    ax.scatter(X, y, c=colors, s=10, alpha=0.6, edgecolors='none')
    ax.set_xlabel('Happiness')
    ax.set_ylabel('CDF')
    ax.grid(True, alpha=0.3)

    # Plot male CDF with confidence interval
    ax.plot(cdf_x, male_cdf, color=palettes.tue_plot[3], label='Male CDF', linewidth=0.8)
    male_lower = np.clip(male_cdf - epsilon_male, 0, 1)
    male_upper = np.clip(male_cdf + epsilon_male, 0, 1)
    ax.fill_between(cdf_x, male_lower, male_upper, alpha=0.3, color=palettes.tue_plot[3])

    # Plot female CDF with confidence interval
    ax.plot(cdf_x, female_cdf, color=palettes.tue_plot[0], label='Female CDF', linewidth=0.8)
    female_lower = np.clip(female_cdf - epsilon_female, 0, 1)
    female_upper = np.clip(female_cdf + epsilon_female, 0, 1)
    ax.fill_between(cdf_x, female_lower, female_upper, alpha=0.3, color=palettes.tue_plot[0])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palettes.tue_plot[3], label='Male'),
        Patch(facecolor=palettes.tue_plot[0], label='Female')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)

    plt.savefig('compact_gender_bias.pdf')



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, '..', '..', 'data', 'newspaper_collection_evaluation_results_20_12_2025.csv')
    main(path)


