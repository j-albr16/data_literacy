import pandas as pd
from tqdm import tqdm
import os
import gc
import numpy as np
import tensorflow as tf
from keras import backend as K
from utils import get_process_args


from models.emotion import EmotionModel, DeepFaceEmotionModel
from models.recognition import RecognitionModel, DeepFaceModel
from models.extraction import (ExtractionModel, 
                              DeepFaceExtractionModel, 
                              extract_face_from_array)
from PIL import Image


def subsample_df(df: pd.DataFrame, start, end):
    if  end < 0 or end > len(df):
        end = len(df)
    elif end <= start:
        raise ValueError("Invalid value for end.")

    return df.iloc[start:end]


def process(
    emotion_model: EmotionModel, 
    recognition_model: RecognitionModel,
    extraction_model: ExtractionModel, 
    articles_csv_path: str,
    omit_tqdm=False,
    start=0,
    end=-1,
    out_name="out.csv",
    batch_size = 64,
    data_dir = 'data',
):
    art_df = pd.read_csv(articles_csv_path)
    # out_path = os.path.join(data_dir, 'out.csv')

    # validate columns
    assert all(c in art_df.columns for c in ['date', 'image_path', 'newspaper']) , 'not all columns present'

    # init results
    art_df = subsample_df(df=art_df, start=start, end=end)
    columns = ['name', 'surname', 'confidence', 'distance', 'date', 'article', 'newspaper', 'dominant_emotion']

    print("*","="*80,"*")
    print(f"Starting recognition/emotion detection for {len(art_df)} images.")
    print(f"\t☺ From: {start} to {len(art_df) if end<=0 else end}.")
    print(f"\t☺ Saving out to: {out_name} - Please make sure the dir exist (else crash).")
    
    results = []
    write_header = not os.path.exists(out_name)
    
    print("*","="*80,"*")


    iterator = tqdm(range(len(art_df))) if not omit_tqdm else range(len(art_df))

    for i in iterator:

        # get article infos
        idx = art_df.index[i]
        article = art_df.loc[idx]
        image_path = os.path.join(data_dir, article['image_path'])
        date = article['date']
        newspaper = article['newspaper']

        # load image as np.array
        try:
            with Image.open(image_path) as img:
                image_arr = np.array(img)
        except Exception as e:
            print(f"Error: {e}")
            continue

        try:
            extracted_face_info = extraction_model(image_arr)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"[{i}/{len(art_df)}] Extracted {len(extracted_face_info)} faces from '{image_path}'.")
        for face_info in extracted_face_info:
            extracted_face = extract_face_from_array(
                    image=image_arr, 
                    face_location=face_info["facial_area"])
            # find politician
            try:
                name, surname, confidence, distance = recognition_model(extracted_face)
                # try to detach from graph.
                confidence = float(confidence)
                distance = float(distance)
            except Exception as e:
                print(f'failed to detect face in {image_path}: {e}')
                continue

            try:
                dominant_emotion, emotions = emotion_model(extracted_face)
            except Exception as e:
                print(f'failed to get emotions for {image_path}: {e}')
                continue

            # unmark data
            del extracted_face 
            # add emotions keys to result
            current_columns = columns + list(emotions.keys())

            #  add entry to results
            emotion_values = [float(v) for v in emotions.values()]
            entry = [name, surname, confidence, distance, date, image_path, newspaper, dominant_emotion]
            entry += emotion_values
            results.append(entry)

        del image_arr
        del extracted_face_info
        if len(results) >= batch_size:
            result = pd.DataFrame(results, columns=current_columns)
            print("[Wrote content to file.]")
            result.to_csv(out_name, mode='a', header=write_header, index=False)
            write_header = False
            
            del result
            results = []
        gc.collect()
        if (i % 100) == 0:
            print("Cleared Keras session")
            K.clear_session()
            tf.compat.v1.reset_default_graph()



    if results:
        result = pd.DataFrame(results, columns=current_columns)
        result.to_csv(out_name, mode='a', header=write_header, index=False)
        del result
        gc.collect()

    print(f"Finished at index {end}") 


if __name__ == "__main__":
    emotion_model = DeepFaceEmotionModel()
    args = get_process_args()
    data_dir = args.data_dir
    recognition_model = DeepFaceModel(
            csv_path=os.path.join(data_dir, args.politician_reference_csv), 
            data_dir=data_dir,
            silent=True,
            detector_backend="retinaface")
    extraction_model = DeepFaceExtractionModel(
            detector_backend="retinaface",
            expand_percentage=20
            )
    article_csv = os.path.join(data_dir, args.article_data_csv)

    end = -1 if args.num_images < 0 else (args.start + args.num_images)

    process(
        emotion_model=emotion_model,
        recognition_model=recognition_model,
        extraction_model=extraction_model,
        articles_csv_path=article_csv,
        start=args.start,
        end=end,
        omit_tqdm=args.omit_tqdm,
        out_name=args.out_name,
        data_dir=data_dir
    )
