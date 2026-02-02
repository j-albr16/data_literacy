import os
from abc import ABC, abstractclassmethod, abstractmethod
from typing import List, Sequence, Tuple

from matplotlib.pylab import ndarray
import numpy as np
import pandas as pd
from deepface import DeepFace

# from utils import normalize_filename


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
    def __call__(self, img_path: str | np.ndarray) -> Tuple[str, str, float, float]:
        """
        accepts and img path
        searches for that in image in the database
        returns (name, surname, distance, confidence)
        """
        pass

class DeepFaceModel(RecognitionModel):

    def __init__(self, 
                 csv_path: str, 
                 data_dir: str, 
                 silent=False,
                 detector_backend="retinaface",
                 model_name="VGG-Face") -> None:
        self.df = pd.read_csv(csv_path)
        self.silent = silent
        self.detector_backend = detector_backend
        self.model_name = model_name
        assert self._is_correct(self.df), f'csv does not contain required columns'

        # append data dir to image paths
        self.db_path = os.path.join(data_dir, self.df.iloc[0]["image_path"].split('/')[0])
        self.df["image_path"] = self.df["image_path"].apply(lambda p: os.path.join(data_dir, p))
        # self.df['image_path'] = self.df['image_path'].apply(normalize_filename)

    def __call__(
            self,
            img_path: str | ndarray,
            k = 5,
            confidence_threshold = 0.,
            distance_threshold = float('inf')
    ) -> Tuple[str, str, float, float]:
        dfs: List[pd.DataFrame] = DeepFace.find(img_path = img_path,
                                                model_name=self.model_name,
                                                db_path=self.db_path,
                                                detector_backend=self.detector_backend,
                                                silent=self.silent)

        assert len(dfs) > 0, 'no result'
        detected_persons = []
        for i, df in enumerate(dfs):

            # only take the k nearest neighbours
            if i >= k:
                break

            if df.empty:
                continue

            first = df.iloc[0]

            # locate politician
            politician = self.df[self.df['image_path'] == first['identity']].iloc[0]

            name = politician['name']
            surname = politician['surname']
            confidence = first['confidence']
            distance = first['distance']

            detected_persons += [(name, surname, confidence, distance)]

        assert len(detected_persons) > 0, 'no detected persons'

        # do majority vote
        surnames = [surname for _, surname, _, _ in detected_persons]
        values, counts = np.unique(surnames, return_counts=True)
        best_name = values[np.argmax(counts)]

        print(f'selected {best_name} as best name from {surnames}')
        
        # get the values for the best name
        name, surname, confidence, distance = detected_persons[surnames.index(best_name)]
        
        # check tresholds
        assert confidence >= confidence_threshold, 'confidence threshold not passed'
        assert distance <= distance_threshold, 'distnace threshold not passed'

        return name, surname, confidence, distance







