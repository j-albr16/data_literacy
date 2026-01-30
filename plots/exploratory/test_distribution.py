import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import copy
import os

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

from utils import add_bins

plt.rcParams.update(figsizes.icml2024_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})
plt.rcParams.update(bundles.icml2024(column='half'))


B = 1000

def main(path, emotion='happy'):
    """
    check happiness bias for the following distribution:
    p(happy|age, others)
    and
    p(happy|gender, others)
    where we check all combinations of others.
    all properties are:
        - popularity (#articles)
        - age
        - newspaper
        - party
        - gender
        - personality
    """

    df = pd.read_csv(path)

    # norm emotions
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotions = df[all_emotions].to_numpy()
    emotions = emotions / np.sum(emotions, axis=-1, keepdims=True)
    df[emotion] = emotions[:, all_emotions.index(emotion)]

    # aggregate age
    df['age'] = 2025 - df['birth']

    # aggregate popularity (number of articles per person)
    df['popularity'] = df.groupby('surname')['surname'].transform('count')

    # add bins for binned values
    popularity_bins = [350, 1000]
    age_bins = [45, 60, 70]
    df['popularity_bin'] = df['popularity'].transform(add_bins(popularity_bins))
    df['age_bin'] = df['age'].transform(add_bins(age_bins))

    # test_configurations = {k: df[k].unique() for k in ['popularity_bin', 'age_bin', 'party', 'newspaper', 'surname', 'gender']}
    test_configurations = {k: df[k].unique() for k in ['popularity_bin', 'age_bin', 'party', 'newspaper', 'gender']}

    def run_test(df_, target='gender', **kwargs):
        # filter df
        try:
            for k, v in kwargs.items():
                if k != target:
                    df_ = df_[df_[k] == v]

                if len(df_) == 0:
                    return
        except Exception:
            print(kwargs)
            assert True

        # run test
        X = df_[emotion].to_numpy()

        if target == 'gender':
            is_true = (df_['gender'] == 'male').to_numpy()
        elif target == 'age_bin':
            is_true = (df_['age_bin'] >= 2).to_numpy()

        # Check for balanced groups - require at least 20% in each group and minimum 10 samples
        n_true = np.sum(is_true)
        n_false = np.sum(~is_true)
        min_group_size = 10
        min_proportion = 0.2

        if n_true < min_group_size or n_false < min_group_size:
            return  # Skip if either group is too small

        proportion_true = n_true / len(is_true)
        if proportion_true < min_proportion or proportion_true > (1 - min_proportion):
            return  # Skip if groups are too imbalanced

        rng = np.random.default_rng(42)
        X_permuted = rng.permuted(np.tile(X, (B,1)), axis=1)

        def t_mean(happiness_values):
            return np.mean(happiness_values[is_true]) - np.mean(happiness_values[~is_true])

        # test: mean difference
        mean_true = t_mean(X)
        mean_dist = np.apply_along_axis(t_mean, 1, X_permuted)
        p_value_mean = np.sum(np.abs(mean_dist) >= np.abs(mean_true)) / B

        return mean_true, mean_dist, p_value_mean, len(df_)


    # Generate all combinations of test configurations
    if not (os.path.exists('tests_age.csv') and os.path.exists('tests_gender.csv')):
        keys = list(test_configurations.keys())
        values = list(test_configurations.values())

        targets = ['gender', 'age_bin']
        results = {k: [] for k in targets}
        for combination in tqdm(product(*values)):
            config = dict(zip(keys, combination))

            for target in targets:
                result = run_test(df, target=target, **config)
                if result is not None:
                    mean_true, mean_dist, p_value_mean, size = result
                    config_clone = copy.copy(config)
                    config_clone['mean_true'] = mean_true
                    config_clone['p_value'] = p_value_mean
                    config_clone['size'] = size
                    results[target].append(config_clone)

        # Convert results to DataFrame for easier analysis
        results_age_df = pd.DataFrame(results['age_bin'])
        results_gender_df = pd.DataFrame(results['gender'])
        print(f"\nTotal configurations tested: {len(results_gender_df)}")

        results_age_df.to_csv('tests_age.csv')
        results_gender_df.to_csv('tests_gender.csv')

    else:
        print('load data')
        results_age_df = pd.read_csv('tests_age.csv')
        results_gender_df = pd.read_csv('tests_gender.csv')

    def plot(ax, df, title):
        pvalues = df['p_value']
        mean = pvalues.mean()
        median = pvalues.median()

        counts, bins = np.histogram(pvalues, bins=30)
        # ax1.step(bins[:-1], counts, where='post', linewidth=1.5, color=palettes.tue_plot[0])
        ax.hist(pvalues, bins=30)
        ax.step(bins[:-1], counts, where='post', linewidth=1.5, color=palettes.tue_plot[0])
        ax.axvline(x=mean, color=palettes.tue_plot[1], linestyle='--', linewidth=1, label=f'mean: {mean:.3f}')
        ax.axvline(x=median, color=palettes.tue_plot[2], linestyle='--', linewidth=1, label=f'median: {median:.3f}')
        # ax.set_title(title)
        ax.set_xlabel('p-value')
        ax.set_ylabel('frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)



    # plot the results
    fig, ax = plt.subplots()
    # fig, ax1 = plt.subplots()

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
    legend_elements = [
        Patch(facecolor=palettes.tue_plot[0], label='age'),
        Patch(facecolor=palettes.tue_plot[1], label='gender')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel('p-value')
    plt.savefig('p_value_distr.pdf')
    
    # plot(ax1, results_gender_df, 'Gender Bias Test')
    # plot(ax2, results_age_df, 'Age Bias Test')

    newspapers = df['newspaper'].unique()
    targets = ['Gender', 'Age']

    plt.savefig('dist_bias_tests.pdf')

    fig, ax3 = plt.subplots()

    results = {'age': [], 'gender': []}
    for j, n in enumerate(newspapers):
        df_ = results_gender_df
        df_ = df_[df_['newspaper'] == n]
        results['gender'].append(df_['p_value'].mean())

        df_ = results_age_df
        df_ = df_[df_['newspaper'] == n]
        results['age'].append(df_['p_value'].mean())

    x = np.arange(len(newspapers)) 
    width = 0.25
    multiplier = 0

    for attribute, measurement in results.items():
        offset = width * multiplier
        rects = ax3.bar(x + offset, measurement, width, label=attribute)
        ax3.bar_label(rects, fmt='%.2f', padding=3, rotation=90, clip_on=True)
        multiplier += 1


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax3.set_ylabel('mean p-value')
    ax3.set_xticks(x + width, newspapers)
    ax3.legend(loc='upper right', ncols=2)
    ax3.set_ylim(0, 1)  # P-values range from 0 to 1
    ax3.set_ylim(top=ax3.get_ylim()[1] * 1.1)
    ax3.grid(True, alpha=0.3)


    plt.savefig('mean_p_value.pdf')

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

    fig, ax4 = plt.subplots()

    ax4.scatter(X, y, c=colors, s=10, alpha=0.6, edgecolors='none')
    ax4.set_xlabel('Happiness')
    ax4.set_ylabel('CDF')
    ax4.grid(True, alpha=0.3)

    # Plot male CDF with confidence interval
    ax4.plot(cdf_x, male_cdf, color=palettes.tue_plot[3], label='Male CDF')
    male_lower = np.clip(male_cdf - epsilon_male, 0, 1)
    male_upper = np.clip(male_cdf + epsilon_male, 0, 1)
    ax4.fill_between(cdf_x, male_lower, male_upper, alpha=0.3, color=palettes.tue_plot[3])

    # Plot female CDF with confidence interval
    ax4.plot(cdf_x, female_cdf, color=palettes.tue_plot[0], label='Female CDF')
    female_lower = np.clip(female_cdf - epsilon_female, 0, 1)
    female_upper = np.clip(female_cdf + epsilon_female, 0, 1)
    ax4.fill_between(cdf_x, female_lower, female_upper, alpha=0.3, color=palettes.tue_plot[0])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palettes.tue_plot[3], label='Male'),
        Patch(facecolor=palettes.tue_plot[0], label='Female')
    ]
    ax4.legend(handles=legend_elements, loc='best', fontsize=8)


    plt.savefig('compact_bias_tests.pdf')


if __name__ == "__main__":
    path = '/home/scrutycs/uni/data_literacy/politicians/data/politicians_results.csv'
    main(path)

