import pandas as pd
import numpy as np
from itertools import chain
import json
from utility import filter_labels

molecules = pd.read_csv("../../../../data/old_features/pyrfume/leffingwell/molecules.csv")
behavior = pd.read_csv("../../../../data/old_features/pyrfume/leffingwell/behavior_sparse.csv")

# Exclude molecules where no CID is known
molecules.drop(molecules[molecules.CID == 0].index, inplace=True)
behavior.drop(behavior[behavior.CID == 0].index, inplace=True)

molecules.drop(columns=["MolecularWeight", "IUPACName", "name", "cas"], inplace=True)
behavior.drop(columns=["IsomericSMILES", "Raw Labels"], inplace=True)

molecules = molecules.set_index("CID")
behavior = behavior.set_index("CID")

new_df = molecules.join(behavior, "CID")
new_df = new_df.rename(columns={"Labels": "Odors"})
new_df["Odors"] = new_df["Odors"].apply(lambda x: json.loads(x.replace("'", '"')))

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

new_df.to_json("../../../../output/old_features/data/pyrfume/leffingwell.json", indent=4)
