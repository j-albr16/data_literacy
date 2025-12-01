import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from PIL import Image
import numpy as np

emotion_colors = ['#ff0000', '#996633', '#ff33cc', '#009933', '#0066ff', '#ffff00', '#666699']

def plot_emotions(ax, emotion):
    emotion_grid = np.zeros((len(emotion), 1, 3))
    for j, (prob, color) in enumerate(zip(emotion, emotion_colors)):
            rgb = np.array([int(color[k:k+2], 16) for k in (1, 3, 5)]) / 255.0
            emotion_grid[j, 0] = rgb * prob + (1 - prob)  # Blend with white
        
    # print(emotion_grid.shape)
    ax.imshow(emotion_grid, aspect='auto', interpolation='nearest')
    ax.set_yticks([])
    ax.set_xticks([])
    

def main(csv_path = 'out.csv', num=20):
    df = pd.read_csv(csv_path)
    df = df.sample(n=num)

    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    images = df['article']
    emotions = df[emotion_labels]
    names= df['surname']
    distances = df['distance']
    confidences = df['confidence']

    n_images = len(images)
    rows = 5
    columns = n_images // rows

    fig, axes = plt.subplots(rows, 2*columns, figsize=(20, 4*rows))

    for i in range(n_images):
        image = images.iloc[i]
        emotion = np.array(emotions.iloc[i])
        emotion = emotion / np.sum(emotion)
        name = names.iloc[i]
        distance = distances.iloc[i]
        confidence = confidences.iloc[i]

        row = i%rows
        column = 2*(i//rows)

        # plot image
        img = Image.open(image)
        axes[row, column].imshow(img)
        axes[row, column].axis('off')
        axes[row, column].set_title(f'{name} - {confidence*100:.2f} % - $||x||^2_2 = {distance}$')

        # plot emotions
        plot_emotions(axes[row, column+1], emotion)

    # Create legend (only once for the whole figure)
    legend_elements = [mpatches.Patch(facecolor=color, label=label) 
                      for color, label in zip(emotion_colors, emotion_labels)]
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.98), fontsize=12,
              title='Emotions', title_fontsize=13)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
