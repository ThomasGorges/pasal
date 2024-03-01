import pandas as pd
import numpy as np
import json
from itertools import chain
from utility import filter_labels

molecules = pd.read_csv("../../../../data/old_features/pyrfume/arctander/molecules.csv")
behavior = pd.read_csv("../../../../data/old_features/pyrfume/arctander/behavior_1_sparse.csv")
identifiers = pd.read_csv("../../../../data/old_features/pyrfume/arctander/identifiers.csv")

# Merge behaviour with identifiers
behavior = behavior.set_index("Stimulus")
identifiers = identifiers.set_index("Stimulus")

annotations = behavior.join(identifiers, "Stimulus")
annotations = annotations.set_index("new_CID")
annotations.index.names = ["CID"]

# Merge molecules with annotations
new_df = molecules.join(annotations, "CID")
new_df = new_df.drop(
    columns=["MolecularWeight", "IUPACName", "ChemicalName", "CAS", "name"]
)
new_df = new_df.set_index("CID")
new_df = new_df.rename(columns={"ChastretteDetails": "Odors"})
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

# new_df.to_csv("../output/pyrfume/arctander.csv")
new_df.to_json("../../../../output/old_features/data/pyrfume/arctander.json", indent=4)
