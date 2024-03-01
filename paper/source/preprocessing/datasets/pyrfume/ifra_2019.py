import pandas as pd
from utility import drop_uncommon_classes, filter_cids

molecules = pd.read_csv("../../../data/pyrfume/ifra_2019/molecules.csv")
behavior = pd.read_csv(
    "../../../data/pyrfume/ifra_2019/behavior.csv", dtype={"Descriptor 1": "str"}
)
stimuli = pd.read_csv("../../../data/pyrfume/ifra_2019/stimuli.csv")

behavior["Descriptor 1"] = behavior["Descriptor 1"].astype(str)

# Exclude other columns
molecules = molecules.set_index("CID")
molecules = molecules[["IsomericSMILES"]]

# Merge descriptors into one column
behavior = behavior.set_index("Stimulus")
stimuli = stimuli.set_index("Stimulus")

labels = behavior.join(stimuli, "Stimulus")
labels = labels.set_index("CID")

labels["Odors"] = labels[
    ["Descriptor 1", "Descriptor 2", "Descriptor 3"]
].values.tolist()

new_df = molecules.join(labels, "CID").drop(
    columns=["Descriptor 1", "Descriptor 2", "Descriptor 3"]
)

new_df = new_df[~new_df.index.duplicated(keep="first")]

new_df = filter_cids(new_df)

new_df.to_json("../../../output/data/pyrfume/ifra_2019_full.json", indent=4)

new_df = drop_uncommon_classes(new_df)

new_df.to_json("../../../output/data/pyrfume/ifra_2019.json", indent=4)
