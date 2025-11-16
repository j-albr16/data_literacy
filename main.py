import argparse

import pandas as pd
from tqdm import tqdm

from emotion import EmotionModel, DeepFaceEmotionModel
from recognition import RecognitionModel, DeepFaceModel
from utils import rename_files


def process(
    emotion_model: EmotionModel, 
    recognition_model: RecognitionModel,
    articles_csv_path: str,
    politician_base_dir = 'party_members' ,
    batch_size = 64,
):
    rename_files(politician_base_dir)
    art_df = pd.read_csv(articles_csv_path)

    # validate columns
    assert all(c in art_df.columns for c in ['date', 'image_path', 'newspaper']) , 'not all columns present'

    # init results
    results = []
    columns = ['name', 'surname', 'confidence', 'distance', 'date', 'article', 'newspaper', 'dominant_emotion']

    for i in tqdm(range(len(art_df))):

        # get article infos
        article = art_df.iloc[i]
        image_path = article['image_path']
        date = article['date']
        newspaper = article['newspaper']

        # find politician
        try:
            name, surname, distance, confidence = recognition_model(image_path)
            dominant_emotion, emotions = emotion_model(image_path)
        except Exception as e:
            print(f'failed to process {image_path}: {e}')
            continue
 
        # add emotions keys to result
        for k in emotions.keys():
            if k not in columns:
                columns.append(k)

        print(emotions.values())
        #  add entry to results
        entry = [name, surname, confidence, distance, date, image_path, newspaper, dominant_emotion]
        entry += emotions.values()
        results.append(entry)

        if i % batch_size == 0:
            result = pd.DataFrame(results, columns=columns)
            result.to_csv('out.csv')


    result = pd.DataFrame(results, columns=columns)
    result.to_csv('out.csv')
    print(result)
   


if __name__ == "__main__":
    emotion_model = DeepFaceEmotionModel()
    recognition_model = DeepFaceModel('bundestag_members_with_paths.csv')
    article_csv = 'politician_data_set/politicians.csv'

    process(
        emotion_model,
        recognition_model,
        article_csv
    )
