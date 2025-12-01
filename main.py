import pandas as pd
from tqdm import tqdm

from emotion import EmotionModel, DeepFaceEmotionModel
from recognition import RecognitionModel, DeepFaceModel
from utils import rename_files
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--politician_reference_csv",
                        help="Path to the .csv file, that contains the paths to the images of the politicians that are being used as reference for classification")
    parser.add_argument("--article_data_csv",
                        help="Directory to the .csv containing the article data. (e.g. image path, newspaper, ...)")
    parser.add_argument("--politician_base_dir", 
                        help="Base directory of the politician dataset. Contains the .csv pointing to all the images")
    parser.add_argument("--num_images",
                        type=int,
                        help="How many images to process (For e.g. debugging.)",
                        default=-1
                        )
    parser.add_argument("--start", type=int)
    parser.add_argument("--out_name", type=str, required=False, default="out.csv")
    parser.add_argument("--omit_tqdm", action="store_true", default=False)
    return parser.parse_args()

def subsample_df(df: pd.DataFrame, start, end):
    if end <= start:
        raise ValueError("Invalid value for end.")
    elif end < 0 or end > len(df):
        end = len(df)

    return df.iloc[start:end]

def init_results(out_path):
    """ Try to read a checkpoint from an already existing result file.
        If none is present, initialize empty.
    """
    try:
        df = pd.read_csv(out_path, index_col=False)
        results = [df.iloc[i].tolist() for i in range(len(df))]
        print(f"\t☺Loaded checkpoint file from '{out_path}'.")
        return results

    except FileNotFoundError:
        print(f"\t☺Tried to read: {out_path} - could not find. Initializing empty df.")
        return []


def process(
    emotion_model: EmotionModel, 
    recognition_model: RecognitionModel,
    articles_csv_path: str,
    omit_tqdm=False,
    start=0,
    end=-1,
    politician_base_dir = 'politicians' ,
    out_name="out.csv",
    batch_size = 64,
):
    rename_files(politician_base_dir)
    art_df = pd.read_csv(articles_csv_path)
    # validate columns
    assert all(c in art_df.columns for c in ['date', 'image_path', 'newspaper']) , 'not all columns present'
    
    art_df = subsample_df(df=art_df, start=start, end=end)

    # init results
    columns = ['name', 'surname', 'confidence', 'distance', 'date', 'article', 'newspaper', 'dominant_emotion']
    
    print("*","="*80,"*")
    print(f"Starting recognition/emotion detection for {len(art_df)} images.")
    print(f"\t☺ From: {start} to {end}.")
    print(f"\t☺ Saving out to: {out_name} - Please make sure the dir exist (else crash).")
    results = init_results(out_name)
    print("*","="*80,"*")
    iterator = tqdm(range(len(art_df))) if not omit_tqdm else range(len(art_df))
    for i in iterator:

        # get article infos
        article = art_df.iloc[i]
        image_path = article['image_path']
        date = article['date']
        newspaper = article['newspaper']
        # find politician
        try:
            name, surname, confidence, distance = recognition_model(image_path)
        except Exception as e:
            print(f'failed to detect face in  {image_path}: {e}')
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
            result.to_csv(out_name, index=False)


    result = pd.DataFrame(results, columns=columns)
    result.to_csv(out_name, index=False)
    print(result)
    print(f"Finished at index {end}") 


if __name__ == "__main__":
    emotion_model = DeepFaceEmotionModel()
    args = get_args()
    recognition_model = DeepFaceModel(args.politician_reference_csv)
    article_csv = args.article_data_csv

    end = -1 if args.num_images < 0 else (args.start + args.num_images)
    process(
        emotion_model,
        recognition_model,
        article_csv,
        start=args.start,
        end=end,
        omit_tqdm=args.omit_tqdm,
        politician_base_dir= args.politician_base_dir,
        out_name=args.out_name,
    )
