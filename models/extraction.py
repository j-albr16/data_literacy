import os
from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
from deepface import DeepFace

class ExtractionModel(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """ Extracts Faces from Images """
        pass


class DeepFaceExtractionModel(ExtractionModel):
    def __init__(self, detector_backend: str, expand_percentage: int) -> None:
        super().__init__()
        self.detector_backend = detector_backend
        self.expand_percentage = expand_percentage

    def __call__(self, img_path: str | np.ndarray) -> List[Dict[str, Any]]:
        extraction_info = DeepFace.extract_faces(
                    img_path=img_path,
                    detector_backend=self.detector_backend,
                    expand_percentage=self.expand_percentage
                    )

        return extraction_info


def extract_face_from_array(image: np.ndarray, face_location: dict):
    x, y, w, h = face_location["x"], face_location["y"], face_location["w"], face_location["h"]
    face = image[y:y+h, x:x+w, :]
    return face
