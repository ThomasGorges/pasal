import pandas as pd
import numpy as np
from itertools import chain
from utility import filter_labels


molecules = pd.read_csv("../../../../data/old_features/pyrfume/ifra/molecules.csv")
behavior = pd.read_csv(
    "../../../../data/old_features/pyrfume/ifra/behavior.csv", dtype={"Descriptor 1": "str"}
)

behavior["Descriptor 1"] = behavior["Descriptor 1"].astype(str)

# Exclude other columns
molecules = molecules.set_index("CID")
molecules = molecules[["IsomericSMILES"]]

# Merge descriptors into one column
behavior = behavior.set_index("CID")
behavior["Odors"] = behavior[
    ["Descriptor 1", "Descriptor 2", "Descriptor 3"]
].values.tolist()

new_df = molecules.join(behavior, "CID").drop(
    columns=["Descriptor 1", "Descriptor 2", "Descriptor 3"]
)

new_df = new_df[~new_df.index.duplicated(keep='first')]

# Count occurrences of odors
odors = pd.Series(list(chain.from_iterable(new_df['Odors'])))
odor_count = odors.value_counts()

odors_to_drop = odor_count[odor_count < 19]
print(f"Dropping {len(odors_to_drop)} classes")

new_df['Odors'] = new_df['Odors'].apply(lambda x: filter_labels(x, odors_to_drop))
new_df['Odors'] = new_df['Odors'].replace(odors_to_drop.tolist(), np.nan)

# Drop molecules where no annotation is given
new_df = new_df[new_df['Odors'].map(lambda x: len(x)) != 0]

new_df.to_json("../../../../output/old_features/data/pyrfume/ifra.json", indent=4)
