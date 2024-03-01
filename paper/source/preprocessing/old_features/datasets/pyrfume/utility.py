import pandas as pd
import numpy as np
from itertools import chain


def filter_labels(labels, filter_list):
    new_labels = []

    for class_name in labels:
        if class_name not in filter_list:
            new_labels.append(class_name)

    return new_labels


def drop_uncommon_classes(df, threshold=20):
    odors = pd.Series(list(chain.from_iterable(df["Odors"])))
    odor_count = odors.value_counts()

    odors_to_drop = odor_count[odor_count < threshold]
    print(
        f"Dropping {len(odors_to_drop)} classes ({len(odor_count) - len(odors_to_drop)} remain)"
    )

    df["Odors"] = df["Odors"].apply(lambda x: filter_labels(x, odors_to_drop))
    df["Odors"] = df["Odors"].replace(odors_to_drop.tolist(), np.nan)

    # Drop molecules where no annotation is given
    df = df[df["Odors"].map(lambda x: len(x)) != 0]

    return df


def filter_cids(df):
    # Drop negative CIDs
    # CIDs are negative when real ID is not known
    df = df[df.index > 0]

    # Drop mixtures ('.' in SMILES)
    df = df[~df['IsomericSMILES'].str.contains('.', regex=False)]

    return df