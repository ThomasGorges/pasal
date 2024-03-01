import os
import pandas as pd
import json


DATASET_FOLDER = "../../output/data/pyrfume/"

pyrfume_classes = {}

filenames = os.listdir(DATASET_FOLDER)
dataset_paths = [os.path.join(DATASET_FOLDER, path) for path in filenames]

for dataset_path in dataset_paths:
    if '_full' in dataset_path:
        continue
    
    df = pd.read_json(dataset_path)

    unique_classes = df["Odors"].explode().unique().astype(str)

    unique_classes = sorted(unique_classes)

    dataset_name = dataset_path.rsplit("/")[-1].replace(".json", "")

    pyrfume_classes[dataset_name] = unique_classes

with open("../../output/preprocessing/pyrfume_classes.json", "w") as f:
    json.dump(pyrfume_classes, f, sort_keys=True, indent=4)
