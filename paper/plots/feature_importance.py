import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib
import matplotlib.pyplot as plt
from joblib import dump

matplotlib.rc('font', size=12)

train_cids = pd.read_csv('../data/dream/TrainSetProcessedAvg.csv').set_index('CID').index.values
val_cids = pd.read_csv('../data/dream/LBs2_cids.csv').set_index('CID').index.values
test_cids = pd.read_csv('../data/dream/GSs2_cids.csv').set_index('CID').index.values

features_df = pd.read_csv('../output/study_results/11082023_fold_999_999/718/features_shared_model_perspective.csv').set_index('CID')
features_df = features_df.loc[[*train_cids]]
features_df = features_df[~features_df.index.duplicated(keep='first')]

labels_df = pd.read_json('../output/preprocessing/fusion.json', orient='index')
labels_df.index.name = 'CID'

labels_df = labels_df.drop(columns=['IsomericSMILES'])
labels_df = labels_df.loc[[*train_cids]]

datasets = labels_df.columns.values.tolist()

feature_importance = {}
models = {}

for dataset in datasets:
    y = labels_df[dataset].dropna()
    cids = y.index.values

    X = features_df.loc[cids]
    
    if dataset == 'DREAM':
        models[dataset] = RandomForestRegressor(random_state=0)
        y = y.tolist()
    else:
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y)
        models[dataset] = RandomForestClassifier(random_state=0)
    
    models[dataset].fit(X, y)
    feature_importance[dataset] = np.array(models[dataset].feature_importances_)

dump(models, '../output/analysis/feature_importance_models.joblib')
dump(feature_importance, '../output/analysis/feature_importances.joblib')

importance_df = pd.DataFrame.from_records(feature_importance)

importance_df = importance_df.rename(columns={
    'dream_mean': 'DREAM mean',
    'dream_std': 'DREAM std',
    'arctander': 'Arctander',
    'ifra_2019': 'IFRA',
    'leffingwell': 'Leffingwell',
    'sigma_2014': 'Sigma'
})

def plot_heatmap(df):
    plt.close()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(df, interpolation='none', aspect='auto', cmap='plasma')
    ax.set_yticks(np.arange(len(df.index)), labels=df.index)
    ax.set_xlabel('Feature index')

    ax.figure.colorbar(im, ax=ax).set_label('Feature importance')

    fig.tight_layout()

    return fig

truncated_df = importance_df.loc[importance_df.sum(axis=1) != 0]
truncated_df = truncated_df.reset_index(drop=True)

sorted_indices = truncated_df.sum(axis=1).sort_values(ascending=False).index.values
sorted_df = truncated_df.reindex(index=sorted_indices).reset_index(drop=True)

fig = plot_heatmap(sorted_df.T)
fig.savefig('../output/plots/fig10.pdf')
