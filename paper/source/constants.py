import numpy as np
import sys
import os
from utility.loader import calculate_num_classes, load_classes, load_cooccurrence_data, calculate_db_indices
from utility.score import read_DREAM_submission
from joblib import load


# https://github.com/NVIDIA/tensorflow-determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Generate an initial seed based on random 4 bytes (should equal size of int)
SEED = int.from_bytes(os.urandom(4), sys.byteorder, signed=False)

NUM_FOLDS = 10
NUM_FEATURES = 1286

DREAM_ODORS = ['INTENSITY/STRENGTH', 'VALENCE/PLEASANTNESS', 'BAKERY', 'SWEET', 'FRUIT', 'FISH', 'GARLIC', 'SPICES',
               'COLD', 'SOUR', 'BURNT', 'ACID', 'WARM', 'MUSKY', 'SWEATY', 'AMMONIA/URINOUS', 'DECAYED', 'WOOD',
               'GRASS', 'FLOWER', 'CHEMICAL']

PYRFUME_ODORS = load_classes('../output/preprocessing/pyrfume_classes.json')
NUM_ODORS = calculate_num_classes(PYRFUME_ODORS)
TOTAL_CLASS_COUNT = np.sum(list(NUM_ODORS.values()))

DATASETS = [
    'dream_mean',
    'dream_std',
    'arctander',
    'ifra_2019',
    'leffingwell',
    'sigma_2014'
]
NUM_DATASETS = len(DATASETS)  # DREAM, arctander, ifra, leffingwell, sigma_2014

DB_START_INDICES, DB_END_INDICES = calculate_db_indices(DATASETS, PYRFUME_ODORS)

DREAM_NUM_CLASSES = 42  # 38 odors (mean & std) + 2 intensity (mean & std) + 2 pleasantness (mean & std)

LBS2_SCALED_FILE_PATH = '../output/data/dream/LBs2_scaled.csv'
GOLD_STANDARD_FILE_PATH = '../data/dream/GSs2_new.txt'

COOCCURRENCE_FILE_PATH = '../output/preprocessing/known_combinations.json'

COOCCURRENCE_DATA = load_cooccurrence_data(COOCCURRENCE_FILE_PATH)

# These are needed in order to know the correct molecule order in the submission files and to calculate performance on validation & test set
GS_CID_ORDER, GS_GROUND_TRUTH = read_DREAM_submission(GOLD_STANDARD_FILE_PATH)
LBS2_CID_ORDER, LBS2_GROUND_TRUTH = read_DREAM_submission(LBS2_SCALED_FILE_PATH)

DREAM_SCALER = load('../output/preprocessing/dream_scaler.joblib')
