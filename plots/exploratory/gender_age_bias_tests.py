import copy
import os
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import add_bins

B = 1000

def bias_test(df_, emotion, target='gender', **kwargs):
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

        # Skip if either group is too small
        if n_true < min_group_size or n_false < min_group_size:
            return  

        # Skip if groups are too imbalanced
        proportion_true = n_true / len(is_true)
        if proportion_true < min_proportion or proportion_true > (1 - min_proportion):
            return  

        rng = np.random.default_rng(42)
        X_permuted = rng.permuted(np.tile(X, (B,1)), axis=1)

        def t_mean(happiness_values):
            return np.mean(happiness_values[is_true]) - np.mean(happiness_values[~is_true])

        # test: mean difference
        mean_true = t_mean(X)
        mean_dist = np.apply_along_axis(t_mean, 1, X_permuted)
        p_value_mean = np.sum(np.abs(mean_dist) >= np.abs(mean_true)) / B

        return mean_true, mean_dist, p_value_mean, len(df_)


def get_biases_test_distribution(path, emotion='happy'):
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

    test_configurations = {k: df[k].unique() for k in ['popularity_bin', 'age_bin', 'party', 'newspaper', 'gender']}

    # try to load data
    if (os.path.exists('tests_age.csv') and os.path.exists('tests_gender.csv')):
        print('load data')
        results_age_df = pd.read_csv('tests_age.csv')
        results_gender_df = pd.read_csv('tests_gender.csv')
        return results_age_df, results_gender_df

    # generate all combinations of test configurations
    keys = list(test_configurations.keys())
    values = list(test_configurations.values())

    targets = ['gender', 'age_bin']
    results = {k: [] for k in targets}
    for combination in tqdm(product(*values)):
        config = dict(zip(keys, combination))

        for target in targets:
            result = bias_test(df, target=target, **config)

            if result is not None:
                mean_true, mean_dist, p_value_mean, size = result
                config_clone = copy.copy(config)
                config_clone['mean_true'] = mean_true
                config_clone['p_value'] = p_value_mean
                config_clone['size'] = size
                results[target].append(config_clone)

    results_age_df = pd.DataFrame(results['age_bin'])
    results_gender_df = pd.DataFrame(results['gender'])

    results_age_df.to_csv('tests_age.csv')
    results_gender_df.to_csv('tests_gender.csv')

    return results_age_df, results_gender_df





