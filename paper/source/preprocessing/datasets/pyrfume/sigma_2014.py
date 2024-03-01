import pandas as pd
import json
from utility import drop_uncommon_classes, filter_cids

molecules = pd.read_csv("../../../data/pyrfume/sigma_2014/molecules.csv")
behavior = pd.read_csv("../../../data/pyrfume/sigma_2014/behavior_sparse.csv")

# Exclude molecules where no CID is known
molecules.drop(molecules[molecules.CID == 0].index, inplace=True)

molecules.drop(columns=["MolecularWeight", "IUPACName", "name"], inplace=True)

molecules = molecules.set_index("CID")

behavior = behavior.rename(columns={"Stimulus": "CID"})
behavior.drop(behavior[behavior.CID == 0].index, inplace=True)
behavior = behavior.set_index("CID")

new_df = molecules.join(behavior, "CID")
new_df = new_df.rename(columns={"descriptors": "Odors"})
new_df["Odors"] = new_df["Odors"].apply(lambda x: json.loads(x.replace("'", '"')))

new_df = new_df[~new_df.index.duplicated(keep="first")]

new_df = filter_cids(new_df)

new_df.to_json("../../../output/data/pyrfume/sigma_2014_full.json", indent=4)

new_df = drop_uncommon_classes(new_df)

new_df.to_json("../../../output/data/pyrfume/sigma_2014.json", indent=4)
