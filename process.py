import pandas as pd
from tqdm import tqdm
import os
from argparse import ArgumentParser
from utils import get_process_args

from models.emotion import EmotionModel, DeepFaceEmotionModel
from models.recognition import RecognitionModel, DeepFaceModel


def subsample_df(df: pd.DataFrame, start, end):
    if  end < 0 or end > len(df):
        end = len(df)
    elif end <= start:
        raise ValueError("Invalid value for end.")

    return df.iloc[start:end]

def init_results(out_path):
    """ Try to read a checkpoint from an already existing result file.
        If none is present, initialize empty.
    """
    try:
        df = pd.read_csv(out_path, index_col=False)
        results = [df.iloc[i].tolist() for i in range(len(df))]
        print(f"\t☺ Loaded checkpoint file from '{out_path}'.")
        return results

    except FileNotFoundError:
        print(f"\t☺ Tried to read: {out_path} - could not find. Initializing empty df.")
        return []


def process(
    emotion_model: EmotionModel, 
    recognition_model: RecognitionModel,
    articles_csv_path: str,
    omit_tqdm=False,
    start=0,
    end=-1,
    out_name="out.csv",
    batch_size = 64,
    data_dir = 'data',
):
    art_df = pd.read_csv(articles_csv_path)
    out_path = os.path.join(data_dir, 'out.csv')

    # validate columns
    assert all(c in art_df.columns for c in ['date', 'image_path', 'newspaper']) , 'not all columns present'

    # init results
    art_df = subsample_df(df=art_df, start=start, end=end)
    columns = ['name', 'surname', 'confidence', 'distance', 'date', 'article', 'newspaper', 'dominant_emotion']

    print("*","="*80,"*")
    print(f"Starting recognition/emotion detection for {len(art_df)} images.")
    print(f"\t☺ From: {start} to {len(art_df) if end<=0 else end}.")
    print(f"\t☺ Saving out to: {out_name} - Please make sure the dir exist (else crash).")
    results = init_results(out_name)
    print("*","="*80,"*")


    iterator = tqdm(range(len(art_df))) if not omit_tqdm else range(len(art_df))

    for i in iterator:

        # get article infos
        article = art_df.iloc[i]
        print(article.tolist())
        image_path = os.path.join(data_dir, article['image_path'])
        date = article['date']
        newspaper = article['newspaper']

        # find politician
        try:
            name, surname, distance, confidence = recognition_model(image_path)
        except Exception as e:
            print(f'failed to detect face in {image_path}: {e}')
            continue

        try:
            dominant_emotion, emotions = emotion_model(image_path)
        except Exception as e:
            print(f'failed to get emotions for {image_path}: {e}')
            continue
 
        # add emotions keys to result
        for k in emotions.keys():
            if k not in columns:
                columns.append(k)

        #  add entry to results
        entry = [name, surname, confidence, distance, date, image_path, newspaper, dominant_emotion]
        entry += emotions.values()
        results.append(entry)

        if i % batch_size == 0:
            result = pd.DataFrame(results, columns=columns)
            result.to_csv(out_path)


    result = pd.DataFrame(results, columns=columns)
    result.to_csv(out_name, index=False)
    print(result)
    print(f"Finished at index {end}") 


if __name__ == "__main__":
    emotion_model = DeepFaceEmotionModel()
    args = get_process_args()
    data_dir = args.data_dir
    recognition_model = DeepFaceModel(os.path.join(data_dir, args.politician_reference_csv), data_dir)
    article_csv = os.path.join(data_dir, args.article_data_csv)

    end = -1 if args.num_images < 0 else (args.start + args.num_images)

    process(
        emotion_model,
        recognition_model,
        article_csv,
        start=args.start,
        end=end,
        omit_tqdm=args.omit_tqdm,
        out_name=args.out_name,
        data_dir=data_dir
    )
