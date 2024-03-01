import pandas as pd
import json
from utility import drop_uncommon_classes, filter_cids

molecules = pd.read_csv("../../../data/pyrfume/arctander/molecules.csv")
behavior = pd.read_csv("../../../data/pyrfume/arctander/behavior_1_sparse.csv")
stimuli = pd.read_csv("../../../data/pyrfume/arctander/stimuli.csv")

# Merge behaviour with identifiers
behavior = behavior.set_index("Stimulus")
stimuli = stimuli.set_index("Stimulus")

annotations = behavior.join(stimuli, "Stimulus")
annotations = annotations.set_index("new_CID")
annotations.index.names = ["CID"]

# Merge molecules with annotations
new_df = molecules.join(annotations, "CID")
new_df = new_df.drop(
    columns=["MolecularWeight", "IUPACName", "ChemicalName", "CAS", "name"]
)
new_df = new_df.set_index("CID")
new_df = new_df.rename(columns={"Labels": "Odors"})
new_df["Odors"] = new_df["Odors"].apply(lambda x: json.loads(x.replace("'", '"')))

new_df = new_df[~new_df.index.duplicated(keep="first")]

new_df = filter_cids(new_df)

new_df.to_json("../../../output/data/pyrfume/arctander_full.json", indent=4)

new_df = drop_uncommon_classes(new_df)

new_df.to_json("../../../output/data/pyrfume/arctander.json", indent=4)
