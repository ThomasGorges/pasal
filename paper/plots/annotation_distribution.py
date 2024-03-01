import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm

matplotlib.rcParams.update({'font.size': 12})

molecules = pd.read_json('../output/preprocessing/fusion.json', orient='index')

train_folds = pd.read_json('../output/preprocessing/train_folds.json', orient='index')
train_cids = train_folds.loc[999]['train_cids']

train_set = molecules.loc[train_cids].drop(columns=['IsomericSMILES', 'dream_mean', 'dream_std'])

distributions = {}

for dataset in train_set.columns:
    partial_df = train_set[dataset].dropna()

    distributions[dataset] = []

    for cid in partial_df.index:
        mol = partial_df.loc[cid]
        
        num_labels = len(mol)
        distributions[dataset].append(num_labels)

fig, axs = plt.subplots(1, 4, figsize=(12, 3.5))

x = 0

for db_name in ['arctander', 'ifra_2019', 'leffingwell', 'sigma_2014']:
    dist = distributions[db_name]
    name_to_display = ''
    if db_name == 'ifra_2019':
        name_to_display = 'IFRA'
    elif db_name == 'leffingwell':
        name_to_display = 'Leffingwell'
    elif db_name == 'sigma_2014':
        name_to_display = 'Sigma'
    elif db_name == 'arctander':
        name_to_display = 'Arctander'
    elif db_name == 'all':
        name_to_display = 'Categorical datasets combined'

    mean, sigma = norm.fit(dist)

    num_bins = 11

    n, bins, patches = axs[x].hist(dist, np.arange(num_bins) + 0.5, range=(1, num_bins), rwidth=0.8, density=True, align='mid')

    axs[x].plot(np.arange(num_bins), norm.pdf(np.arange(num_bins), mean, sigma), 'r--')

    axs[x].set_xlim(0)

    axs[x].set_title(f'{name_to_display}\n' + rf'($\mu={mean:.2f}$, $\sigma={sigma:.2f}$)')
    axs[x].set_xlabel('Number of annotations')
    axs[x].set_ylabel('Density')
    axs[x].set_xticks(np.arange(1, num_bins))

    x += 1

fig.tight_layout()
plt.savefig('../output/plots/fig3.pdf', dpi=1200)
plt.show()
