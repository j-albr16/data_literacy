import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update(figsizes.icml2022_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})

B = 1000

def main(
    path,
    newspaper = 'freitag',
    party = 'all'
):
    print(party, newspaper)

    # get all parties that
    df = pd.read_csv(path)

    # norm emotions
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotions = df[all_emotions].to_numpy()
    emotions = emotions / np.sum(emotions, axis=-1, keepdims=True)
    df['happy'] = emotions[:, all_emotions.index('happy')]

    rng = np.random.default_rng(42)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    if newspaper != 'all':
        df = df[df['newspaper'] == newspaper]

    if party != 'all':
        df = df[df['party'] == party]

    assert len(df) > 0, f'nespaper={newspaper}, party={party}'

    df_ = df[['gender', 'happy']]
    X = df_['happy'].to_numpy()
    is_male = (df_['gender'] == 'male').to_numpy()

    def t_mean(happiness_values):
        return np.mean(happiness_values[is_male]) - np.mean(happiness_values[~is_male])

    def t_top(happiness_values):
        return np.min(happiness_values[is_male]) - np.min(happiness_values[~is_male])

    sorting = np.argsort(X)
    def t_proportions(group, k):
        top_k = sorting[:k]
        return np.sum(group[top_k]) / k

    X_permuted = rng.permuted(np.tile(X, (B,1)), axis=1)

    # Test 1: Mean difference
    mean_true = t_mean(X)
    mean_dist = np.apply_along_axis(t_mean, 1, X_permuted)
    p_value_mean = np.sum(np.abs(mean_dist) >= np.abs(mean_true)) / B

    axs[0].hist(mean_dist, bins=50, alpha=0.7, edgecolor='black')
    axs[0].axvline(mean_true, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Obs (p={p_value_mean:.4f})')
    axs[0].set_title(f'party={party} - newspaper={newspaper} - Mean Diff')
    axs[0].set_xlabel('Mean Difference (M - F)')
    axs[0].set_ylabel('Frequency')
    axs[0].legend(fontsize=8)

    # Scatter plot showing happiness distribution by gender
    rng_scatter = np.random.default_rng(42)
    y = rng_scatter.normal(loc=0.5, scale=0.1, size=X.shape)

    # calculate male and female cdf
    cdf_x = np.linspace(0, 1, 1000)
    males = X[is_male]
    females = X[~is_male]
    male_cdf = np.array([len(males[males < x]) / len(males) for x in cdf_x])
    female_cdf = np.array([len(females[females < x]) / len(females) for x in cdf_x])
    epsilon_male = np.sqrt((1/(2*len(males))) * np.log(2 / 0.05) )
    epsilon_female = np.sqrt((1/(2*len(females))) * np.log(2 / 0.05) )

    # Use tue_plot colors for male/female
    colors = [palettes.tue_plot[0] if male else palettes.tue_plot[3] for male in is_male]

    axs[1].scatter(X, y, c=colors, s=10, alpha=0.6, edgecolors='none')
    axs[1].set_title(f'party={party} - newspaper={newspaper} - Mean Diff')
    axs[1].set_xlabel('Happiness')
    axs[1].set_ylabel('CDF / Random Jitter')
    axs[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Plot male CDF with confidence interval
    axs[1].plot(cdf_x, male_cdf, color=palettes.tue_plot[3], label='Male CDF')
    male_lower = np.clip(male_cdf - epsilon_male, 0, 1)
    male_upper = np.clip(male_cdf + epsilon_male, 0, 1)
    axs[1].fill_between(cdf_x, male_lower, male_upper, alpha=0.3, color=palettes.tue_plot[3])

    # Plot female CDF with confidence interval
    axs[1].plot(cdf_x, female_cdf, color=palettes.tue_plot[0], label='Female CDF')
    female_lower = np.clip(female_cdf - epsilon_female, 0, 1)
    female_upper = np.clip(female_cdf + epsilon_female, 0, 1)
    axs[1].fill_between(cdf_x, female_lower, female_upper, alpha=0.3, color=palettes.tue_plot[0])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palettes.tue_plot[3], label='Male'),
        Patch(facecolor=palettes.tue_plot[0], label='Female')
    ]
    axs[1].legend(handles=legend_elements, loc='best', fontsize=8)


    plt.tight_layout()
    plt.savefig(f'data/plots/gender_bias_{newspaper}_{party}_happyness.png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    path = '/home/scrutycs/uni/data_literacy/politicians/data/politicians_results.csv'
    for n in ['sz', 'freitag', 'compact']:
        for party in ['spd', 'gruenen', 'afd', 'union', 'linke']:
            main(path, newspaper=n, party=party)

