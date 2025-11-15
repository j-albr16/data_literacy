from abc import ABC, abstractclassmethod, abstractmethod
from typing import List, Sequence, Tuple

import pandas as pd
from deepface import DeepFace
from tensorflow.python.ops.math_ops import Reciprocal
from utils import normalize_filename


class RecognitionModel(ABC):

    @abstractmethod
    def __init__(self, csv_path: str) -> None:
        """
        path to the csv file containing politicians and the image path
        """
        pass

    def _is_correct(self, df: pd.DataFrame):
        return all(c in df.columns for c in ["name","surname","party","image_path"])


    @abstractmethod
    def __call__(self, img_path: str) -> Tuple[str, str, float, float]:
        """
        accepts and img path
        searches for that in image in the database
        returns (name, surname, distance, confidence)
        """
        pass

class DeepFaceModel(RecognitionModel):

    def __init__(self, csv_path: str) -> None:
        self.df = pd.read_csv(csv_path)
        assert self._is_correct(self.df), f'csv does not contain required columns'

        self.db_path = self.df.iloc[0]["image_path"].split('/')[0]
        self.df['image_path'] = self.df['image_path'].apply(normalize_filename)

    def __call__(self, img_path: str) -> Tuple[str, str, float, float]:
        dfs: List[pd.DataFrame] = DeepFace.find(img_path = img_path, db_path=self.db_path)

        assert len(dfs) > 0, 'no result'
        df: pd.DataFrame = dfs[0]
        assert not df.empty, 'empty dataframe'
        first = df.iloc[0]

        # locate politician
        politician = self.df[self.df['image_path'] == first['identity']].iloc[0]

        name = politician['name']
        surname = politician['surname']
        confidence = first['confidence']
        distance = first['distance']

        return name, surname, confidence, distance





