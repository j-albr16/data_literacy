import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import jit
from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

from utils import add_datetime

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update(figsizes.icml2022_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})

Q_FACTOR = 1e1
R_FACTOR = 1e3

def KalmanFilter(A, Q, H, R):
    @jit
    def predict(m, P, t, std):
        m_ = A @ m
        P_ = A @ P @ A.T + Q(t, std)
        return m_, P_

    @jit
    def update(m_, P_, z, t, std):
        S = H @ P_ @ H.T + R(t, std)
        K = jnp.linalg.solve(S.T, H @ P_).T
        m = m_ + K @ (z - H @ m_)
        P = P_ - K @ H @ P_
        # Ensure P doesn't collapse to zero (numerical stability)
        P = P + 1e-8 * jnp.eye(P.shape[0])
        return m, P

    @jit
    def smooth(ms, Ps, m, P, t, std):
        m_ = A @ m
        P_ = A @ P @ A.T + Q(t, std)
        # Add regularization to prevent singular matrix when P collapses to zero
        P_ = P_ + 1e-6 * jnp.eye(P_.shape[0])
        G = jnp.linalg.solve(P_, A @ P).T
        ms = m + G @ (ms - m_)
        Ps = P + G @ (Ps - P) @ G.T
        return ms, Ps

    return predict, update, smooth

def smooth(emotion_logits, std_logits, days):
    # std_logits = jnp.sqrt(jnp.clip(jnp.log(std_logits), min=0))
    dim = emotion_logits.shape[1]
    A = jnp.eye(dim)
    Q = lambda delta_t, std:  std * Q_FACTOR * delta_t * jnp.eye(dim)
    H = jnp.eye(dim) 
    R = lambda delta_t, std: std * delta_t * R_FACTOR * jnp.eye(dim)

    predict, update, smooth = KalmanFilter(A, Q, H, R)
    m0 = jnp.ones(dim) / dim      # uniform emotions prior
    P0 = 1 * jnp.eye(dim)
    m = m0
    P = P0

    means = []
    covs = []
    days_list = []
    last_day = 0

    # Forward pass (filtering)
    for i, day in enumerate(days):
        delta_t = day - last_day
        m_, P_ = predict(m, P, delta_t, jnp.diag(std_logits[i, :]))
        x = emotion_logits[i, :]

        m, P = m_, P_
        if not jnp.isnan(x).all():
            m, P = update(m_, P_, x, delta_t, jnp.diag(std_logits[i, :]))

        if jnp.isnan(m).any():
            print(f"NaN detected at iteration {i}, day {day}")
            print(f"  m_: {m_}, has NaN: {jnp.isnan(m_).any()}")
            print(f"  m: {m}, has NaN: {jnp.isnan(m).any()}")
            print(f"  P diagonal: {jnp.diag(P)}")
            break

        means.append(m)
        covs.append(P)
        days_list.append(day)
        last_day = day

    # Backward pass (smoothing)
    means = jnp.array(means)
    covs = jnp.array(covs)
    print(f"Forward pass complete. means has NaN: {jnp.isnan(means).any()}, covs has NaN: {jnp.isnan(covs).any()}")
    smooth_means = [means[-1]]
    smooth_covs = [covs[-1]]

    for i in range(len(means) - 2, -1, -1):
        delta_t = days_list[i+1] - days_list[i]
        ms, Ps = smooth(smooth_means[0], smooth_covs[0], means[i], covs[i], delta_t, jnp.diag(std_logits[i, :]))

        if jnp.isnan(ms).any():
            print(f"NaN in backward pass at i={i}")
            print(f"  delta_t: {delta_t}")
            print(f"  covs[i] diagonal: {jnp.diag(covs[i])}")

        smooth_means.insert(0, ms)
        smooth_covs.insert(0, Ps)

    smooth_means = jnp.array(smooth_means)
    print("smooth_means before softmax:", smooth_means.shape, "has NaN:", jnp.isnan(smooth_means).any(), "has inf:", jnp.isinf(smooth_means).any())
    smooth_means = jax.nn.softmax(smooth_means)
    print("smooth_means after softmax:", smooth_means.shape, "has NaN:", jnp.isnan(smooth_means).any())
    smooth_covs = jnp.diagonal(jnp.array(smooth_covs), axis1=-1, axis2=-2)
    smooth_covs = jax.nn.softmax(smooth_covs)
    return smooth_means, smooth_covs


def timeline(
        path: str,
        surname = 'merz',
        # emotion = 'disgust',
        emotion = 'disgust',
        events = [('2025-10-14', 'Stadtbild Debate'), ('2025-02-23', 'Federal Elections'), ('2025-05-06', 'Chancelor Elections')],
        newspaper = 'all',
        # newspaper = 'freitag',
):

    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    df = pd.read_csv(path)
    df = add_datetime(df)

    if newspaper != 'all':
        df = df[df['newspaper'] == newspaper].reset_index(drop=True)
    # filter for the surname
    if surname != 'all':
        df = df[df['surname'] == surname].reset_index(drop=True)

    # sort by date
    df = df.sort_values('days_since_min').reset_index(drop=True)
    min_date = pd.to_datetime(df['datetime'].iloc[0])

    daily_stats = df.groupby('days_since_min')[all_emotions].agg(['mean', 'std'])
    df_mean = daily_stats.xs('mean', axis=1, level=1)
    df_std = daily_stats.xs('std', axis=1, level=1).fillna(1)

    # Handle NaN and zero values to prevent log(0) = -inf
    df_mean = df_mean.fillna(1e-10)  # Replace NaN with small value
    df_mean = df_mean.clip(lower=1e-10)  # Ensure no zeros

    days = df_mean.index.to_numpy()

    # smooth over all the emotions
    mean_profile = df_mean[all_emotions].to_numpy()
    mean_profile = mean_profile / np.sum(mean_profile, axis=-1, keepdims=True)
    std_profile = df_std[all_emotions].to_numpy()
    # std_profile = std_profile / np.sum(std_profile, axis=-1, keepdims=True)

    print("emotion_logits:", mean_profile.shape, "has NaN:", jnp.isnan(mean_profile).any(), "has inf:", jnp.isinf(mean_profile).any())
    print("std_logits:", std_profile.shape, "has NaN:", jnp.isnan(std_profile).any())
    # smooth_means, smooth_covs = emotion_logits, std_logits
    smooth_means, smooth_covs = smooth(mean_profile, std_profile, days)
    # smooth_covs = emotion_logits

    emotion_index = all_emotions.index(emotion)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize= (8,6))

    # Get mean and standard deviation for the emotion
    # mean = smooth_means[:, emotion_index]
    mean = smooth_means
    std = jnp.sqrt(smooth_covs)

    mean_profile = jax.nn.softmax(mean_profile)
    std_profile = jax.nn.softmax(std_profile)

    # Set x-axis ticks every 60 days, but show as dates
    max_days = max(days)
    tick_positions = np.arange(0, max_days + 1, 60)

    # Convert tick positions (days) back to dates
    tick_dates = [min_date + pd.Timedelta(days=int(d)) for d in tick_positions]
    tick_labels = [date.strftime('%Y-%m-%d') for date in tick_dates]

    # Apply tick settings to both axes
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_xlabel('Date')

    # Plot the mean line
    for i, emotion in enumerate(all_emotions):
        ax1.plot(days, mean[:, i], label = emotion)
        ax2.plot(days, mean_profile[:, i], label = emotion)
        # ax1.fill_between(
        #         days,
        #     mean[:, i] - 0.05 * std[:, i],
        #     mean[:, i] + 0.05 * std[:, i],
        #         alpha=0.3
        #     )
        
        # ax2.fill_between(
        #     days,
        #     emotion_logits[:, emotion_index] - std_logits[:, emotion_index],
        #     emotion_logits[:, emotion_index] + std_logits[:, emotion_index],
        #     alpha=0.3
        # )
    ax1.legend()
    ax2.legend()

    # Add event markers
    for event_date, event_label in events:
        # Convert event date to days_since_min (ensure both are timezone-naive)
        event_datetime = pd.to_datetime(event_date).tz_localize(None)
        min_date_naive = pd.to_datetime(min_date).tz_localize(None) if hasattr(min_date, 'tz') and min_date.tz else min_date
        event_days = int((event_datetime - min_date_naive).total_seconds() / 86400)

        # Add vertical line on both axes
        for ax in [ax1, ax2]:
            ax.axvline(event_days, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.text(event_days, ax.get_ylim()[1], event_label,
                   rotation=90, verticalalignment='top',
                   horizontalalignment='right', fontsize=8)

    # Add shading for the covariance (mean ± std)

    ax1.set_title('smoothed')
    ax2.set_title('raw')

    fig.suptitle(f'{surname} Emotions')
    plt.savefig(f'{surname}_emotions.png', dpi=300)









if __name__ == "__main__":

    path = '/home/scrutycs/uni/data_literacy/politicians/data/politicians_results.csv'
    politicians = ['merz', 'merkel', 'trump', 'putin']
    for p in politicians:
        timeline(path, surname=p)
