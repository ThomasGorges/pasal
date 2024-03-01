import pandas as pd
import json
from utility import drop_uncommon_classes, filter_cids

molecules = pd.read_csv("../../../data/pyrfume/leffingwell/molecules.csv")
behavior = pd.read_csv("../../../data/pyrfume/leffingwell/behavior_sparse.csv")
stimuli = pd.read_csv("../../../data/pyrfume/leffingwell/stimuli.csv")

# Exclude molecules where no CID is known
molecules.drop(molecules[molecules.CID == 0].index, inplace=True)

molecules.drop(columns=["MolecularWeight", "IUPACName", "name"], inplace=True)
behavior.drop(columns=["IsomericSMILES", "Raw Labels"], inplace=True)
stimuli.drop(
    columns=["name", "cas", "MolecularWeight", "IUPACName", "IsomericSMILES"],
    inplace=True,
)

molecules = molecules.set_index("CID")
behavior = behavior.set_index("Stimulus")
stimuli = stimuli.set_index("Stimulus")

labels = behavior.join(stimuli, "Stimulus")
labels.drop(labels[labels.CID == 0].index, inplace=True)
labels = labels.set_index("CID")

new_df = molecules.join(labels, "CID")
new_df = new_df.rename(columns={"Labels": "Odors"})
new_df["Odors"] = new_df["Odors"].apply(lambda x: json.loads(x.replace("'", '"')))

new_df = new_df[~new_df.index.duplicated(keep="first")]

new_df = filter_cids(new_df)

new_df.to_json("../../../output/data/pyrfume/leffingwell_full.json", indent=4)

new_df = drop_uncommon_classes(new_df)

new_df.to_json("../../../output/data/pyrfume/leffingwell.json", indent=4)
