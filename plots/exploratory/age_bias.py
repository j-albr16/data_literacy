import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tueplots import bundles, cycler, figsizes
import tueplots
from tueplots.constants.color import palettes

plt.rcParams.update(bundles.icml2024())
plt.rcParams.update(figsizes.icml2024_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})

B = 1000



def main(
        path,
        emotion = 'happy'
):
    df = pd.read_csv(path)

    # norm emotions
    all_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotions = df[all_emotions].to_numpy()
    emotions = emotions / np.sum(emotions, axis=-1, keepdims=True)
    df[emotion] = emotions[:, all_emotions.index(emotion)]

    # aggregate age
    df['age'] = 2025 - df['birth']

    # apply age bins
    age_bins_ = [60, 70]
    def get_age_bin(age):
        if age <= age_bins_[0]:
            return 0
        elif age <= age_bins_[1]:
            return 1
        else:
            return 2


    df['age_bin'] = df['age'].aggregate(get_age_bin)

    # make the scatter plot
    X = df[emotion].to_numpy()
    age_bins = df['age_bin']
    rng_scatter = np.random.default_rng(42)
    y = rng_scatter.normal(loc=0.5, scale=0.1, size=X.shape)
    cdf_x = np.linspace(0, 1, 1000)


    fig, (ax, ax2) = plt.subplots(1, 2)

    for i in range(max(age_bins) + 1):
        age_bin = X[age_bins == i] 
        print(i)
        age_bin_cdf = np.array([len(age_bin[age_bin < x]) / len(age_bin) for x in cdf_x])
        epsilon = np.sqrt((1/(2*len(age_bin))) * np.log(2 / 0.05) )

        ax.plot(cdf_x, age_bin_cdf, color=palettes.tue_plot[i])
        lower = np.clip(age_bin_cdf - epsilon, 0, 1)
        upper = np.clip(age_bin_cdf + epsilon, 0, 1)
        ax.fill_between(cdf_x, lower, upper, alpha=0.3, color=palettes.tue_plot[i])

    ax2.hist(df['age'])

    colors = [palettes.tue_plot[i] for i in age_bins]
    ax.scatter(X, y, c=colors, s=10, alpha=0.6, edgecolors='none')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palettes.tue_plot[0], label=f'age $\leq$ {age_bins_[0]}'),
        Patch(facecolor=palettes.tue_plot[1], label=f'{age_bins_[0]} $\leq$ age $<$ {age_bins_[1]}'),
        Patch(facecolor=palettes.tue_plot[2], label=f'age $>$ {age_bins_[1]}')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)

    # Add titles and labels
    fig.suptitle(f'Age Bias in {emotion} Emotion', fontsize=12, y=0.98)
    ax.set_xlabel(f'{emotion} Score')
    ax.set_ylabel('Cumulative Distribution Function')
    ax.set_title('CDF by Age Group')

    ax2.set_xlabel('Age')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Age Distribution')

    plt.tight_layout()
    plt.savefig(f'data/plots/age_bias_{emotion}.png', dpi=300)



if __name__ == "__main__":

    path = '/home/scrutycs/uni/data_literacy/politicians/data/politicians_results.csv'
    for emotion in ['angry', 'happy', 'neutral', 'fear']:
        main(path, emotion=emotion)
    
