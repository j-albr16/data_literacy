import pickle

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tueplots import cycler
from tueplots import figsizes
from tueplots.constants.color import palettes
from tueplots import bundles


plt.rcParams.update(bundles.icml2022())
plt.rcParams.update(figsizes.icml2022_full())
plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
plt.rcParams.update({"figure.dpi": 350})




def faces(pkl_path: str):
    # load embeddings
    with open(pkl_path, 'rb') as f:
        embeddings = pickle.load(f)

    # print(embeddings[0].keys())
    print(embeddings[0]['identity'])

    names = [e['identity'].split('/')[-2].split('_')[-1] for e in embeddings]
    unique_strings = list(set(names))
    string_to_int = {s: i for i, s in enumerate(unique_strings)}

    # Apply mapping
    names_id = [string_to_int[s] for s in names]


    embeddings = np.stack([e['embedding'] for e in embeddings])



    # Compute both t-SNE and PCA embeddings
    tsne_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random', perplexity=3).fit_transform(embeddings)
    pca = PCA(n_components=2)
    pca_embedded = pca.fit_transform(embeddings)

    # Create color map for politicians - need 40 distinct colors
    num_politicians = len(unique_strings)
    # Use Pastel1/Set3 for softer colors, or desaturate hsv colors
    import colorsys

    # Generate softer, less intense colors
    colors = []
    for i in range(num_politicians):
        hue = i / num_politicians
        # Convert HSV to RGB with reduced saturation and increased value for pastel effect
        rgb = colorsys.hsv_to_rgb(hue, 0.5, 0.9)  # saturation=0.5, value=0.9 for softer colors
        colors.append(rgb)

    # Map each point to its politician's color
    point_colors = [colors[names_id[i]] for i in range(len(names_id))]

    # Create figure with 2 subplots (vertically stacked)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    # Helper function to plot embeddings
    def plot_embedding(ax, embedded, title):
        ax.scatter(embedded[:,0], embedded[:,1], c=point_colors, s=20, alpha=0.7)

        # Add text labels at the center of each politician's cluster
        for i, politician in enumerate(unique_strings):
            # Find all points belonging to this politician
            mask = np.array(names_id) == i
            if mask.any():
                # Find the median point (the actual point closest to the coordinate-wise median)
                politician_points = embedded[mask]
                median_coords = np.median(politician_points, axis=0)
                # Find the point closest to the median
                distances = np.linalg.norm(politician_points - median_coords, axis=1)
                median_point = politician_points[np.argmin(distances)]

                ax.text(median_point[0], median_point[1], politician, fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.1, edgecolor='none'))
        ax.set_title(title)

    # Plot t-SNE
    plot_embedding(ax1, tsne_embedded, 't-SNE')

    # Plot PCA
    plot_embedding(ax2, pca_embedded, 'PCA')

    # Add title at the top of the entire figure
    fig.suptitle('Dim reduced face embeddings of DeepFace', fontsize=14)
    plt.tight_layout()
    plt.savefig('face.png')
    plt.show()


if __name__ == "__main__":
    pkl_path = '/home/scrutycs/uni/data_literacy/politicians/data/politicians/ds_model_vggface_detector_retinaface_aligned_normalization_base_expand_0.pkl'
    faces(pkl_path)



