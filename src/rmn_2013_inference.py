import cv2
from rmn import RMN
from typing import Tuple, Type, List, Dict
import pprint
from pathlib import Path
import pandas as pd
import numpy as np
pp = pprint.PrettyPrinter(indent=4)

PROCESSED_DATA_BASE_PATH = Path('data/processed')
ATTRIBUTE_CSV_PATH =  PROCESSED_DATA_BASE_PATH / 'attributes.csv'
IMAGE_BASE_PATH = PROCESSED_DATA_BASE_PATH / 'images'

ModelResults = Tuple[str, float, List[Dict[str, float]]]
# Exmple:
# (   'fear',
#     0.38351646065711975,
#     [   {'angry': 0.33417871594429016},
#         {'disgust': 0.007862000726163387},
#         {'fear': 0.38351646065711975},
#         {'happy': 0.2105707973241806},
#         {'sad': 0.011881764978170395},
#         {'surprise': 0.0030304063111543655},
#         {'neutral': 0.04895983263850212}])

def predict_emotion_for_image(m: RMN, fn: str):
    path = IMAGE_BASE_PATH / fn
    im = cv2.imread(str(path))
    results = m.detect_emotion_for_single_face_image(im)
    return fn, *results

if __name__ == "__main__":
    m = RMN()

    attributes = pd.read_csv(ATTRIBUTE_CSV_PATH)
    filenames = attributes['filename']
    results: List[ModelResults] = []

    for i, fn in filenames.items():
        r = predict_emotion_for_image(m, fn)
        results.append(r)

    results_df = pd.DataFrame(results, columns=('filename', 'prediction', 'prediction_confidence', 'full_labels'))
    results_df.to_csv('results/rmn_2013_results.csv', index=False)

    
