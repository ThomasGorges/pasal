import xgboost as xgb
import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV


train_cids = pd.read_csv('../../output/data/dream/TrainSetProcessed.csv').set_index('CID').index.unique().values
val_cids = pd.read_csv('../../output/data/dream/LBs2Processed.csv').set_index('CID').index.values

train_and_val_cids = np.concatenate([train_cids, val_cids])

# Use unscaled DREAM ground truth data
dream_scaler = load("../../output/preprocessing/dream_scaler.joblib")

labels_df = pd.read_json('../../output/preprocessing/fusion.json', orient='index')
labels_df.index.names = ['CID']
labels_df['dream'] = labels_df['dream_mean'] + labels_df['dream_std']
labels_df = labels_df.loc[train_and_val_cids]
labels_df = labels_df.filter(['dream']).dropna()

y = {}
for cid in labels_df.index:
    scaled_y = labels_df.loc[cid]['dream']
    reordered_y = np.ravel([scaled_y[:21], scaled_y[21:]], order='F')
    unscaled_y = dream_scaler.inverse_transform(reordered_y.reshape(1, -1))[0]
    
    y[cid] = np.concatenate([unscaled_y[0::2], unscaled_y[1::2]]).tolist()

y = pd.DataFrame.from_dict(y, orient='index')
y.index.names = ['CID']

y = np.array(y.values)

cids = pd.read_json('../../output/preprocessing/old_features/fusion.json', orient='index').index.values

# physicochemical_features
mordred_df = pd.read_csv('../../output/preprocessing/old_features/mordred.csv').set_index('CID')
padel_df = pd.read_csv('../../output/preprocessing/old_features/padel_features.csv')
padel_df['CID'] = cids
padel_df = padel_df.set_index('CID').drop(columns=['Name'])

merged_df = pd.concat([padel_df, mordred_df], axis=1)

# Drop duplicate columns
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

dirty_columns = merged_df.select_dtypes(include=['object']).columns.values

# Delete strings
for col in dirty_columns:
    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
    merged_df[col] = merged_df[col].replace(np.nan, 0)

merged_df = merged_df.replace([np.inf, -np.inf], np.nan)

map4_df = pd.read_csv('../../output/preprocessing/old_features/map4_fingerprints.csv').set_index('CID')

features_df = pd.concat([merged_df, map4_df], axis=1)
features_df = features_df.loc[train_and_val_cids]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)
X_scaled = pd.DataFrame(X_scaled, index=features_df.index, columns=scaler.get_feature_names_out())

selector = RFECV(
    estimator=xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=1, random_state=0, n_estimators=10),
    step=0.01,
    n_jobs=-1,
    scoring='r2',
    verbose=2
)
selector.fit(X_scaled, y)

feature_indices = selector.get_support()
feature_names = X_scaled.columns.values[feature_indices].tolist()

feature_names_df = pd.DataFrame.from_dict({'FEATURE_NAME': feature_names})
feature_names_df.to_csv(f'../../output/preprocessing/feature_selection/important_features.csv')

dump(selector, f'../../output/preprocessing/feature_selection/selector.joblib')
dump(scaler, f'../../output/preprocessing/feature_selection/scaler.joblib')
