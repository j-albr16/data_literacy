from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

from deepface import DeepFace


class EmotionModel(ABC):

    @abstractmethod
    def __call__(self, img_path: str | np.ndarray) -> Tuple[str, dict]:
        """
        takes an image path of the image to be analyzed
        returns:
            - dominant emotion
            - dictionary with emotion p dist
        """
        pass

class DeepFaceEmotionModel(EmotionModel):

    def __call__(self, img_path: str | np.ndarray) -> Tuple[str, dict]:
        result = DeepFace.analyze(img_path, actions=['emotion'],
                                  enforce_detection=True,
                                  detector_backend='retinaface'
                                  )

        assert result, 'no result'
        assert len(result) > 0, 'empty result'

        entry = result[0]

        return str(entry['dominant_emotion']), dict(entry['emotion'])






