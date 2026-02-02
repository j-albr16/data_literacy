import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

from kalman import KalmanFilter, smooth
from utils import add_datetime

plt.rcParams.update(figsizes.icml2024_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})
plt.rcParams.update(bundles.icml2024(column='half'))

def timeline(
        path,
        surname,
        events = [
            ('2025-02-23', 'Federal Elections'),
            ('2025-05-06', 'Chancelor Elections'),
            ("2024-09-06", "Collapse Ampel"),
            ("2025-10-20", "Stadtbild Debate"),
        ],
        newspaper = 'sz',
):
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    df = pd.read_csv(path)

    # confidence at least 75%
    df = df[df['confidence'] > 75]

    # add datetime
    df = add_datetime(df)

    # filter newspapers
    if newspaper != 'all':
        df = df[df['newspaper'] == newspaper].reset_index(drop=True)

    # filter for the surname
    if surname != 'all':
        df = df[df['surname'] == surname].reset_index(drop=True)

    assert len(df) > 0, 'dataframe is not empty'

    # sort by date
    df = df.sort_values('days_since_min').reset_index(drop=True)
    min_date = pd.to_datetime(df['datetime'].iloc[0])

    # get mean emotions per day. Also store the stds for smoothing
    daily_stats = df.groupby('days_since_min')[all_emotions].agg(['mean', 'std'])
    df_mean = daily_stats.xs('mean', axis=1, level=1)
    df_std = daily_stats.xs('std', axis=1, level=1).fillna(1)

    # handle NaN and zero
    df_mean = df_mean.fillna(1e-10)  # Replace NaN with small value
    df_mean = df_mean.clip(lower=1e-10)  # Ensure no zeros

    days = df_mean.index.to_numpy()

    # get numpy emotions and stds
    mean_profile = df_mean[all_emotions].to_numpy()
    mean_profile = np.log(mean_profile / np.sum(mean_profile, axis=-1, keepdims=True))
    std_profile = df_std[all_emotions].to_numpy()

    # smooth
    smooth_means, smooth_covs = smooth(mean_profile, std_profile, days)

    fig, ax1 = plt.subplots()

    # Get mean and standard deviation for the emotion
    # mean = smooth_means[:, emotion_index]
    mean = smooth_means
    std = jnp.sqrt(smooth_covs)

    mean_profile = jax.nn.softmax(mean_profile)
    std_profile = jax.nn.softmax(std_profile)

    # Set x-axis ticks every 60 days, but show as dates
    max_days = max(days)
    days_offset = 230
    tick_positions = np.arange(days_offset, max_days + 1, 150)
    days = days[days_offset:]
    mean = mean[days_offset:, :]
    std = std[days_offset:, :]

    # Convert tick positions (days) back to dates
    tick_dates = [min_date + pd.Timedelta(days=int(d)) for d in tick_positions]
    tick_labels = [date.strftime('%Y-%m-%d') for date in tick_dates]

    # Apply tick settings to both axes
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax1.set_xlabel('Date')

    # Plot the mean line
    for i, emotion in enumerate(['happy', 'angry', 'fear', 'neutral']):
        i = all_emotions.index(emotion)
        ax1.plot(days, mean[:, i], label = emotion, linewidth=0.9, alpha=0.9)
        ax1.fill_between(
                days,
                mean[:, i] - 0.15 * std[:, i],
                mean[:, i] + 0.15 * std[:, i],
                alpha=0.2
            )
    ax1.legend(loc='upper left')

    # Add event markers
    for event_date, event_label in events:
        # Convert event date to days_since_min (ensure both are timezone-naive)
        event_datetime = pd.to_datetime(event_date).tz_localize(None)
        min_date_naive = pd.to_datetime(min_date).tz_localize(None) if hasattr(min_date, 'tz') and min_date.tz else min_date
        event_days = int((event_datetime - min_date_naive).total_seconds() / 86400)

        # Add vertical line on both axes
        ax1.axvline(event_days, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.text(event_days, ax1.get_ylim()[1], event_label,
               rotation=90, verticalalignment='top',
               horizontalalignment='right', fontsize=8)

    print('storing result')
    plt.savefig(f'{surname}_emotions.pdf')



if __name__ == "__main__":

    path = '/home/scrutycs/uni/data_literacy/politicians/data/timeline.csv'
    politicians = ['merz', 'merkel', 'trump', 'putin']
    timeline(path, surname='merz')
