import pandas as pd
import numpy as np
import os
from utility import PubChem
from sklearn.preprocessing import MinMaxScaler
from joblib import dump


def read_dream_datasets(folder_path):
    dataframes = []

    filenames = [
        "GSs2_newProcessed.csv",
        "LBs2Processed.csv",
        "TrainSetProcessedAvg.csv",
    ]

    for filename in filenames:
        df = pd.read_csv(os.path.join(folder_path, filename), index_col="CID")

        class_names = df.columns

        df = df[df.columns].apply(lambda x: list(x), axis=1)

        df = df.drop(columns=class_names)
        df = df.replace(np.nan, None)

        dataframes.append(df)

    scaler_values = []
    val_df = dataframes[1]
    train_df = dataframes[2]
    
    for cid in val_df.index.values:
        scaler_values.append(val_df.loc[cid])

    for cid in train_df.index.values:
        scaler_values.append(train_df.loc[cid])

    dream_scaler = MinMaxScaler((0.0, 1.0))
    dream_scaler = dream_scaler.fit(
        scaler_values
    )  # Fit on training and validation data

    dream_df = pd.concat(dataframes).to_frame()
    dream_df = dream_df.rename(columns={0: "dream"})

    dream_mean_df = {}
    dream_std_df = {}

    for cid in dream_df.index.values:
        values = dream_df.loc[cid].to_numpy().tolist()

        values = dream_scaler.transform(values)[0]

        dream_mean_df[cid] = {"dream_mean": values[0::2]}

        dream_std_df[cid] = {"dream_std": values[1::2]}

    dream_mean_df = pd.DataFrame.from_dict(dream_mean_df, orient="index")
    dream_mean_df.index.names = ["CID"]

    dream_std_df = pd.DataFrame.from_dict(dream_std_df, orient="index")
    dream_std_df.index.names = ["CID"]

    return dream_scaler, dream_mean_df, dream_std_df


def read_pyrfume_datasets(folder_path):
    datasets = {}

    db_filenames = os.listdir(folder_path)

    print("Pyrfume datasets:")

    for db_filename in db_filenames:
        db_name = db_filename.replace(".json", "")

        if '_full' in db_name:
            continue

        df = pd.read_json(os.path.join(folder_path, db_filename))
        df.index.names = ["CID"]

        df = df.rename(columns={"Odors": db_name})

        datasets[db_name] = df

        print(f"- {db_name} (N = {df.shape[0]})")

    return datasets


pubchem = PubChem()

dream_scaler, dream_mean_df, dream_std_df = read_dream_datasets("../../output/data/dream/")
pyrfume_datasets = read_pyrfume_datasets("../../output/data/pyrfume/")

dump(dream_scaler, "../../output/preprocessing/dream_scaler.joblib")

dream_test_cids = (
    pd.read_csv("../../output/data/dream/GSs2_newProcessed.csv").set_index("CID").index.values
)

pyrfume_datasets["dream_mean"] = dream_mean_df
pyrfume_datasets["dream_std"] = dream_std_df

fusioned_df = None
for db_name in pyrfume_datasets.keys():
    df_slim = pyrfume_datasets[db_name].copy()
    if "IsomericSMILES" in df_slim.columns:
        df_slim.drop(columns=["IsomericSMILES"], inplace=True)

    if fusioned_df is None:
        fusioned_df = df_slim
    else:
        fusioned_df = fusioned_df.join(df_slim, how="outer")


# SMILES need to be added to the merged dataset
# Search for SMILES notations for all molecules in datasets
# If notations are missing, retrieve them from PubChem
cid_smiles = {}

for cid in fusioned_df.index.values:
    smiles_found = False

    for db_name in pyrfume_datasets.keys():
        ds = pyrfume_datasets[db_name]

        if cid in ds.index and "IsomericSMILES" in ds.columns:
            smiles = ds["IsomericSMILES"].loc[cid]

            cid_smiles[cid] = smiles
            smiles_found = True

    if not smiles_found:
        smiles = pubchem.fetch_smiles_from_cid(cid)
        if smiles:
            cid_smiles[cid] = smiles
    
# Some SMILES changed over time from PubChem
# For reproducibility we set them constant
cid_smiles[5780] = 'OC[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO'
cid_smiles[31209] = 'CCCCCC(=CC1=CC=CC=C1)C=O'
    
# Search for mixtures and drop them, since feature calculation can not process them correctly
cids_to_drop = []

for cid, smiles in cid_smiles.items():
    if '.' in smiles:
        cids_to_drop.append(cid)

print(f'Dropping {len(cids_to_drop)} mixtures')
fusioned_df = fusioned_df.drop(index=cids_to_drop)

# Append SMILES to merged dataframe
fusioned_df["IsomericSMILES"] = fusioned_df.index.map(cid_smiles)

num_unique_molecules = len(fusioned_df.index.unique())
print(f"Number of unique molecules: {num_unique_molecules}")

fusioned_df = fusioned_df.apply(lambda x: x.dropna().to_dict(), axis=1)
fusioned_df.to_json("../../output/preprocessing/fusion.json", orient="index", indent=4)
