import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

from utils import add_datetime
from kalman import smooth

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update(figsizes.icml2022_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})

ALL_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def aggregate_time_periods(dominant_emotions, probabilities, days, min_date, aggregation='daily', data_masks=None):
    """
    Aggregate dominant emotions and probabilities over time periods.

    Args:
        dominant_emotions: array of shape (num_targets, num_days) with emotion indices
        probabilities: array of shape (num_targets, num_days) with probability values
        days: array of day indices
        min_date: pd.Timestamp of the first day
        aggregation: 'daily', 'weekly', or 'monthly'
        data_masks: array of shape (num_targets, num_days) indicating which days have actual data

    Returns:
        agg_emotions, agg_probs, time_labels, agg_masks
    """
    if aggregation == 'daily':
        return dominant_emotions, probabilities, days, data_masks

    # Convert days to dates
    dates = pd.to_datetime([min_date + pd.Timedelta(days=int(d)) for d in days])

    if aggregation == 'weekly':
        # Group by week
        periods = dates.to_period('W')
        period_labels = [f"W{p.week}" for p in periods.unique()]
    elif aggregation == 'monthly':
        # Group by month
        periods = dates.to_period('M')
        period_labels = [p.strftime('%b %Y') for p in periods.unique()]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Aggregate for each target
    num_targets = dominant_emotions.shape[0]
    unique_periods = periods.unique()
    agg_emotions = np.zeros((num_targets, len(unique_periods)), dtype=int)
    agg_probs = np.zeros((num_targets, len(unique_periods)))
    agg_masks = np.zeros((num_targets, len(unique_periods)), dtype=bool)

    for i, period in enumerate(unique_periods):
        mask = periods == period
        for target_idx in range(num_targets):
            # Get most common emotion in this period for this target
            period_emotions = dominant_emotions[target_idx, mask]
            period_probs = probabilities[target_idx, mask]
            period_has_data = data_masks[target_idx, mask]

            # Check if there's any actual data in this period
            if not period_has_data.any():
                agg_masks[target_idx, i] = False
                continue

            agg_masks[target_idx, i] = True

            # Only consider days with actual data
            period_emotions = period_emotions[period_has_data]
            period_probs = period_probs[period_has_data]

            # Find the emotion with highest average probability in this period
            unique_emotions = np.unique(period_emotions)
            best_emotion = unique_emotions[0]
            best_avg_prob = 0

            for emotion in unique_emotions:
                emotion_mask = period_emotions == emotion
                avg_prob = period_probs[emotion_mask].mean()
                if avg_prob > best_avg_prob:
                    best_avg_prob = avg_prob
                    best_emotion = emotion

            agg_emotions[target_idx, i] = best_emotion
            agg_probs[target_idx, i] = best_avg_prob

    return agg_emotions, agg_probs, period_labels, agg_masks


def get_profile(
    df, days
):
    """
    Get emotion profile for the dataframe, aligned with the provided days array.

    Args:
        df: DataFrame with emotion data
        days: Array of day indices to align the profile with

    Returns:
        mean_profile, std_profile, has_data_mask: Arrays of shape (len(days), num_emotions) and bool mask
    """
    df = df.sort_values('days_since_min').reset_index(drop=True)

    daily_stats = df.groupby('days_since_min')[ALL_EMOTIONS].agg(['mean', 'std'])
    df_mean = daily_stats.xs('mean', axis=1, level=1)
    df_std = daily_stats.xs('std', axis=1, level=1).fillna(1)

    # Create DataFrames aligned with the global days array
    # Reindex to include all days, filling missing days with NaN
    df_mean_aligned = df_mean.reindex(days)
    df_std = df_std.reindex(days, fill_value=1)

    # Track which days have actual data (not NaN)
    has_data_mask = ~df_mean_aligned.isnull().all(axis=1).to_numpy()

    # Handle NaN and zero values to prevent log(0) = -inf
    df_mean_aligned = df_mean_aligned.fillna(1e-10)  # Replace NaN with small value
    df_mean_aligned = df_mean_aligned.clip(lower=1e-10)  # Ensure no zeros

    # smooth over all the emotions
    mean_profile = df_mean_aligned[ALL_EMOTIONS].to_numpy()
    mean_profile = mean_profile / np.sum(mean_profile, axis=-1, keepdims=True)
    std_profile = df_std[ALL_EMOTIONS].to_numpy()
    return mean_profile, std_profile, has_data_mask



def get_timeline_data(
    path: str,
    selector='newspaper',   # party | newspaper | surname
    surname=None,       # only when surname is selected in selector
    newspaper=None,     # all | sz | ... -> only when party or person is selected
    aggregation='daily',  # 'daily', 'weekly', or 'monthly'
    events = [('2025-10-14', 'Stadtbild Debate'), ('2025-02-23', 'Federal Elections'), ('2025-05-06', 'Chancelor Elections')],
):
    df = pd.read_csv(path)
    df = add_datetime(df)

    if newspaper != 'all' and selector != 'newspaper':
        df = df[df['newspaper'] == newspaper].reset_index(drop=True)
    if surname != 'all' and selector == 'person':
        df = df[df['surname'] == surname].reset_index(drop=True)

    assert len(df) > 0, 'something is wrong. dataframe is empty'

    # get min date and days
    df = df.sort_values('days_since_min').reset_index(drop=True)
    min_date = pd.to_datetime(df['datetime'].iloc[0])
    daily_stats = df.groupby('days_since_min')[ALL_EMOTIONS].agg(['mean', 'std'])
    df_mean = daily_stats.xs('mean', axis=1, level=1)
    days = df_mean.index.to_numpy()

    targets = df[selector].unique()
    results = []
    data_masks = []

    for target in targets:
        # Filter for this specific target
        target_df = df[df[selector] == target].reset_index(drop=True)
        mean_profile, std_profile, has_data = get_profile(target_df, days)
        # smooth_means = mean_profile
        smooth_means, smooth_covs = smooth(mean_profile, std_profile, days)
        results.append(smooth_means)
        data_masks.append(has_data)

    results = np.array(results)
    data_masks = np.array(data_masks)  # shape: (num_targets, num_days)

    # Get the dominant emotion indices and probabilities
    dominant_emotions = np.argmax(results, axis=-1)  # shape: (num_targets, num_days)
    probabilities = np.take_along_axis(results, dominant_emotions[:, :, np.newaxis], axis=-1).squeeze(-1)

    # Aggregate time periods if needed
    agg_emotions, agg_probs, time_labels, agg_masks = aggregate_time_periods(
        dominant_emotions, probabilities, days, min_date, aggregation, data_masks
    )

    # Plot the result
    plot_emotion_heatmap(agg_emotions, agg_probs, targets, time_labels, aggregation, selector, agg_masks)


def plot_emotion_heatmap(dominant_emotions, probabilities, targets, time_labels, aggregation, selector='Target', data_masks=None):
    """
    Create a heatmap showing dominant emotions with color-coded emotions and intensity-based probabilities.

    Args:
        dominant_emotions: array of shape (num_targets, num_periods) with emotion indices
        probabilities: array of shape (num_targets, num_periods) with probability values
        targets: list of target names
        time_labels: list of time period labels or array of day indices
        aggregation: 'daily', 'weekly', or 'monthly'
        selector: name of what the targets represent (e.g., 'newspaper', 'party', 'surname')
        data_masks: array of shape (num_targets, num_periods) indicating which periods have actual data
    """
    num_targets, num_periods = dominant_emotions.shape

    # Create RGBA image
    img = np.zeros((num_targets, num_periods, 4))

    # Fill in colors based on dominant emotion and probability
    emotion_color_list = palettes.tue_plot

    for i in range(num_targets):
        for j in range(num_periods):
            # Check if there's actual data for this cell
            if data_masks is not None and not data_masks[i, j]:
                # Set to black with full opacity for missing data
                img[i, j] = [0, 0, 0, 1]
                continue

            emotion_idx = dominant_emotions[i, j]
            prob = probabilities[i, j]

            # Get RGB color for this emotion
            [r, g, b] = emotion_color_list[emotion_idx]

            # Set RGBA with alpha based on probability (use full opacity for stronger colors)
            img[i, j] = [r, g, b, 1.0]  # Use full opacity, adjust brightness based on prob
            # Scale the RGB values by probability for intensity
            img[i, j, :3] *= (0.3 + 0.7 * prob)  # Scale between 30% and 100% brightness

    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(6, num_targets * 0.5)))

    # Display the image
    ax.imshow(img, aspect='auto', interpolation='nearest')

    # Add grid lines to create margins between pixels
    ax.set_xticks(np.arange(-0.5, num_periods, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_targets, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    # Set y-axis (targets)
    ax.set_yticks(range(num_targets))
    ax.set_yticklabels(targets)
    ax.set_ylabel(selector.capitalize())

    # Set x-axis (time)
    if aggregation == 'daily':
        # Show fewer x-ticks for daily data
        step = max(1, num_periods // 20)
        ax.set_xticks(range(0, num_periods, step))
        ax.set_xticklabels([f"Day {int(time_labels[i])}" for i in range(0, num_periods, step)], rotation=45, ha='right')
    else:
        # Show all labels for weekly/monthly
        step = max(1, len(time_labels) // 15)
        ax.set_xticks(range(0, len(time_labels), step))
        ax.set_xticklabels([time_labels[i] for i in range(0, len(time_labels), step)], rotation=45, ha='right')

    ax.set_xlabel('Time Period')
    ax.set_title(f'Dominant Emotions Over Time ({aggregation.capitalize()})')

    # Create legend
    legend_elements = [
        Patch(facecolor= palettes.tue_plot[ALL_EMOTIONS.index(emotion)], label=emotion.capitalize())
        for emotion in ALL_EMOTIONS
    ]
    legend_elements.append(Patch(facecolor='black', label='No Data'))
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              title='Emotions', frameon=True)

    plt.tight_layout()
    plt.savefig(f'time_heatmap_{selector}.png', dpi=300)
    plt.show()


def main():
    path = '/home/scrutycs/uni/data_literacy/politicians/data/politicians_results.csv'

    # Options for aggregation: 'daily', 'weekly', 'monthly'
    # Options for selector: 'newspaper', 'party', 'surname'
    get_timeline_data(
        path,
        selector='newspaper',
        aggregation='weekly'  # Change to 'daily', 'weekly', or 'monthly'
    )

if __name__ == "__main__":
    # print(palettes.tue_plot)
    # assert False
    main()
