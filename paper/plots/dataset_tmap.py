import pandas as pd
import matplotlib.pyplot as plt
from mhfp.encoder import MHFPEncoder
import tmap

import random
import numpy as np

random.seed(0)
np.random.seed(0)

datasets_df = pd.read_json('../output/preprocessing/fusion.json', orient='index')

pred_df = pd.read_csv('../output/study_results/11082023_fold_999_999/718/prediction_GSs2.txt', delimiter='\t').set_index('oID')
dream_test_cids = sorted(pred_df.index.unique())

dream_val_cids = sorted(pd.read_csv('../output/study_results/11082023_fold_999_999/718/prediction_LBs2.txt', delimiter='\t').set_index('oID').index.unique())

mhfp_encoder = MHFPEncoder()

fingerprints = []
labels = []
smiles_list = []
markers = []

for cid in datasets_df.index:
    if cid in dream_test_cids:
        labels.append("test")
    elif cid in dream_val_cids:
        labels.append("val")
    else:
        labels.append("train")
        
    smiles = datasets_df.loc[cid]['IsomericSMILES']
    smiles_list.append(smiles)
    
    fp = mhfp_encoder.encode(smiles)
    
    fingerprints.append(tmap.VectorUint(fp))

lf = tmap.LSHForest()
lf.batch_add(fingerprints)
lf.index()

x, y, s, t, _ = tmap.layout_from_lsh_forest(lf)

fig, ax = plt.subplots(figsize=(6, 6))

train_x = []
train_y = []

val_x = []
val_y = []

test_x = []
test_y = []

idx_to_skip = []

for i, l in enumerate(labels):
    # Remove outliers
    if y[i] > 0.49:
        idx_to_skip.append(i)
        continue

    if l == "train":
        train_x.append(x[i])
        train_y.append(y[i])
    elif l == "val":
        val_x.append(x[i])
        val_y.append(y[i])
    else:
        test_x.append(x[i])
        test_y.append(y[i])


# Plot training molecules first
train_scatter = ax.scatter(train_x, train_y, c="#1f77b4", s=20, marker='o')
    
# Plot validation set
val_scatter = ax.scatter(val_x, val_y, c="#59d453", s=50, marker='X')

# Plot test set
test_scatter = ax.scatter(test_x, test_y, c="#fc7b12", s=50, marker='X')

for source_idx, target_idx in zip(s, t):
    if source_idx in idx_to_skip or target_idx in idx_to_skip:
        continue

    x0 = x[source_idx]
    y0 = y[source_idx]

    x1 = x[target_idx]
    y1 = y[target_idx]

    ax.plot([x0, x1], [y0, y1], color='black', linewidth=1)

ax.legend([train_scatter, val_scatter, test_scatter], ['Training', 'Validation', 'Test'], title='Molecule set assignment', loc='lower center', bbox_to_anchor=(0.5, -0.12), ncols=3, fancybox=False, shadow=True)

plt.axis('off')
plt.tight_layout()

plt.savefig('../output/plots/fig6.pdf')
