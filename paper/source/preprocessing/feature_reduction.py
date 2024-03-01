import warnings

warnings.simplefilter(action="ignore")

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump


df = pd.read_csv('../../output/preprocessing/features_merged.csv').set_index('CID')
test_cids = np.unique(pd.read_csv('../../data/dream/GSs2_new.txt', delimiter='\t')['oID'].values.tolist())

# Remove invalid data
mixed_dtype_cols = df.drop(index=test_cids).select_dtypes(include=['object']).columns.values

# Delete strings
for col in mixed_dtype_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

raw_important_features = pd.read_csv("../../output/preprocessing/feature_selection/important_features.csv")["FEATURE_NAME"].tolist()

important_features = []
for col in raw_important_features:
    if 'MAP4_' in col:
        important_features.append(col.replace('MAP4_', 'foldedAP2_'))
    else:
        important_features.append(col)

df = df.loc[:, ~df.columns.duplicated()]

df = df.filter(important_features)

# Replace too high gmin values with NaN instead of dropping it
if 'gmin' in df.columns:
    df.loc[df['gmin'] > 10, 'gmin'] = np.nan

train_cids = pd.read_json("../../output/preprocessing/train_folds.json", orient="index").loc[999]['train_cids']
test_cids = pd.read_json("../../output/preprocessing/test_set.json", orient="index").index.values
val_cids = pd.read_json("../../output/preprocessing/train_folds.json", orient="index").loc[999]['val_cids']

train_and_val_cids = [*train_cids] + [*val_cids]

df = df.replace([np.inf, -np.inf], np.nan)

train_df = df[df.index.isin(train_cids)]
val_df = df[df.index.isin(val_cids)]
test_df = df[df.index.isin(test_cids)]

# Replace CIDs with the ones from the dataframes
# Reason: Order in dataframes might be different
train_cids = train_df.index.values
test_cids = test_df.index.values
val_cids = val_df.index.values

column_names = train_df.columns.values

# Impute missing values
imputer = SimpleImputer(strategy='mean')
imputer.fit(df[df.index.isin(train_and_val_cids)])
train_df = imputer.transform(train_df)
val_df = imputer.transform(val_df)
test_df = imputer.transform(test_df)
dump(imputer, "../../output/preprocessing/imputer.joblib")

# Scale data to N(0,1)
scaler = StandardScaler()
scaler.fit(df[df.index.isin(train_and_val_cids)])
train_df = scaler.transform(train_df)
val_df = scaler.transform(val_df)
test_df = scaler.transform(test_df)
dump(scaler, "../../output/preprocessing/scaler.joblib")

# Reconstruct pandas dataframes
train_df = pd.DataFrame.from_records(train_df).add_prefix("FEATURE_")
train_df["CID"] = train_cids
train_df = train_df.set_index("CID")

val_df = pd.DataFrame.from_records(val_df).add_prefix("FEATURE_")
val_df["CID"] = val_cids
val_df = val_df.set_index("CID")

test_df = pd.DataFrame.from_records(test_df).add_prefix("FEATURE_")
test_df["CID"] = test_cids
test_df = test_df.set_index("CID")

merged_df = pd.concat([
    train_df,
    val_df,
    test_df
])

merged_df.to_csv("../../output/preprocessing/reduced_features.csv")

#print(merged_df)
print(f"Number of features: {merged_df.shape[1]}")
