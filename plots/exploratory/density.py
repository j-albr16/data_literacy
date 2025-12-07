import datetime
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
from plots.exploratory.utils import add_datetime
from tueplots import cycler
from tueplots import figsizes
from tueplots.constants.color import palettes
from tueplots import bundles

from utils import add_datetime

plt.rcParams.update(bundles.icml2022())
plt.rcParams.update(figsizes.icml2022_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})


def my_kde_bandwidth(obj, fac=1./5):

    """We use Scott's Rule, multiplied by a constant factor."""

    return np.power(obj.n, -1./(obj.d+4)) * fac


def plot_density(path: str):
    df = pd.read_csv(path)
    # df = df[df['surname'] == 'merz']
    df = df[["newspaper", "date"]]
    df = add_datetime(df)

    
    newspapers = df["newspaper"].unique()

    # Create a larger figure
    # plt.figure(figsize=(5, 3))

    for newspaper in newspapers:
        newspaper_df = df[df["newspaper"] == newspaper]
        days = newspaper_df["days_since_min"].to_numpy()

        if len(days) > 1:  # Need at least 2 points for KDE
            kde = stats.gaussian_kde(days, bw_method=partial(my_kde_bandwidth, fac=0.35))

            # Create smooth x-axis for plotting
            x_range = np.linspace(days.min(), days.max(), 500)
            kde_values = kde(x_range)

            # Scale KDE by count to show actual frequencies, not normalized density
            kde_values_scaled = kde_values * len(days)

            line = plt.plot(x_range, kde_values_scaled, label=newspaper)[0]
            plt.fill_between(x_range, kde_values_scaled, alpha=0.3, color=line.get_color())

    # Set x-axis ticks every 500 days, but show as dates
    max_days = df['days_since_min'].max()
    tick_positions = np.arange(0, max_days + 1, 60)

    # Convert tick positions (days) back to dates
    tick_dates = [min_date + pd.Timedelta(days=int(d)) for d in tick_positions]
    tick_labels = [date.strftime('%Y-%m-%d') for date in tick_dates]

    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.legend()
    # plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.title('Publishing frequency KDE estimate')
    plt.savefig('density.png')
    # plt.show()


if __name__ == "__main__":
    path = '/home/scrutycs/uni/data_literacy/politicians/data/politicians_results.csv'
    plot_density(
        path
    )



