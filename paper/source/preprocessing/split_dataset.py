import numpy as np
import pandas as pd


NUM_FOLDS = 10

dream_train_cids = (
    pd.read_csv("../../output/data/dream/TrainSetProcessed.csv").set_index("CID").index.values
)
dream_val_cids = (
    pd.read_csv("../../output/data/dream/LBs2Processed.csv").set_index("CID").index.values
)
dream_test_cids = (
    pd.read_csv("../../output/data/dream/GSs2_newProcessed.csv").set_index("CID").index.values
)

dream_cids = [*dream_train_cids, *dream_val_cids, *dream_test_cids]

cids_to_skip = dream_cids

df = pd.read_json("../../output/preprocessing/fusion.json", orient="index")
df.index.name = "CID"

train_val_cids = []
test_cids = dream_test_cids

for cid in df.index.values:
    if cid not in test_cids:
        train_val_cids.append(cid)

train_val_df = df[df.index.isin(train_val_cids)]
test_df = df[df.index.isin(test_cids)]

print(
    f"Train and validation set size: {len(train_val_df)} ({(len(train_val_df)/len(df.index.values)*100.0):.2f}%)"
)
print(
    f"Testset size: {len(test_df)} ({(len(test_df)/len(df.index.values)*100.0):.2f}%)"
)

train_val_df = train_val_df.apply(lambda x: x.dropna().to_dict(), axis=1)
test_df = test_df.apply(lambda x: x.dropna().to_dict(), axis=1)

train_val_df.to_json("../../output/train_val_set.json", orient="index", indent=4)
test_df.to_json("../../output/test_set.json", orient="index", indent=4)

dream_train_val_cids = df[df.index.isin(train_val_cids)]['dream_mean'].dropna().index.values

train_no_dream_cids = []
for cid in df.index.values:
    if cid not in test_cids and cid not in dream_train_val_cids:
        train_no_dream_cids.append(cid)

np.random.seed(0)
np.random.shuffle(dream_train_val_cids)
dream_val_folds = np.array_split(dream_train_val_cids, NUM_FOLDS)

train_folds = {}
for i in range(NUM_FOLDS):
    fold_val_cids = dream_val_folds[i]

    fold_train_cids = []

    for cid in train_val_cids:
        if cid not in fold_val_cids:
            fold_train_cids.append(cid)

    train_folds[i] = {
        'train_cids': fold_train_cids,
        'val_cids': fold_val_cids
    }

train_folds_df = pd.DataFrame.from_dict(train_folds, orient='index')
train_folds_df.index.names = ['fold']

orig_split_train_cids = []
for cid in train_val_cids:
    if cid not in dream_val_cids:
        orig_split_train_cids.append(cid)

train_folds_df.loc[999] = {
    'train_cids': orig_split_train_cids,
    'val_cids': dream_val_cids.tolist()
}

train_folds_df.to_json('../../output/preprocessing/train_folds.json', orient='index', indent=4)
