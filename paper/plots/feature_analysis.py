from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from joblib import dump

matplotlib.rc('font', size=13)

def dim_red(use_lef, use_ifra, save_file, save_model, features_path):
    # load all classes
    classes = pd.read_json("../output/preprocessing/pyrfume_classes.json", orient="index")

    # filter leffingwell classes
    leffingwell = classes.loc['leffingwell'].dropna().tolist()
    leffingwell = [x.lower() for x in leffingwell]
    leffingwell = sorted(leffingwell)


    #filter ifra classes
    ifra = classes.loc['ifra_2019'].dropna().tolist()
    ifra = [x.lower() for x in ifra]
    ifra = sorted(ifra)

    if use_lef:
        common_classes = sorted(leffingwell)
    elif use_ifra:
        common_classes = sorted(ifra)
    else:
        common_classes = list(set(ifra + leffingwell))
        common_classes = sorted(common_classes)


    ######## load and standardize features ########################


    features_df = pd.read_csv(features_path)
    features_df = features_df.drop_duplicates(subset=['CID'])
    features_df = features_df.set_index("CID")
    features_cids = features_df.index.values

    #standardize data
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features_df)

    ############ load labels ###########################

    labels_df = pd.read_json("../output/preprocessing/fusion.json", orient="index")
    labels_df.index.names = ["CID"]

    datasets_to_drop = ["IsomericSMILES", "dream_mean", "dream_std", "sigma_2014", "arctander"]

    if use_lef and not use_ifra:
        datasets_to_drop.append("ifra_2019")
    elif use_ifra and not use_lef:
        datasets_to_drop.append('leffingwell')

    labels_df = labels_df.drop(columns=datasets_to_drop).dropna(how='all')

    labels_tsne = []
    labels_cid_order = []
    features_tsne = []
    features_cid_order = []

    for cid in labels_df.index.values:
        mol = labels_df.loc[cid]
        mol = mol.dropna()
        
        new_label = []

        for ds_name in mol.index.values:
            classes = mol.loc[ds_name]
            for i in range(len(classes)):
                classes[i] = classes[i].lower()
            
            for common_class in common_classes:
                if common_class in classes and common_class not in new_label:
                    name = common_class

                    labels_tsne.append(name)
                    labels_cid_order.append(cid)
                    
                    if cid not in features_cid_order:
                        features_tsne.append(features_std[np.where(features_cids == cid)[0][0]])
                        features_cid_order.append(cid)

    tsne = TSNE(n_components=2, random_state=0, perplexity=150, n_iter=1000)
    features = tsne.fit_transform(np.array(features_tsne))

    df_data = []
    for i in range(len(labels_tsne)):
        query_cid = labels_cid_order[i]

        df_data.append([labels_tsne[i], *features[features_cid_order.index(query_cid)].tolist()])

    dump(tsne, save_model)

    x_name = "TSNE 0"
    y_name = "TSNE 1"

    new_df = pd.DataFrame(df_data, columns=["odor", x_name, y_name])

    new_df.to_csv(save_file)


dim_red(True, False, "../output/analysis/leffingwell.csv", "../output/analysis/leffingwell.joblib", "../output/study_results/11082023_fold_999_999/718/features_leffingwell_perspective.csv")

dim_red(False, True, '../output/analysis/ifra.csv', '../output/analysis/ifra.joblib', "../output/study_results/11082023_fold_999_999/718/features_ifra_2019_perspective.csv")

dim_red(False, False, '../output/analysis/shared_model.csv', '../output/analysis/shared_model.joblib', "../output/study_results/11082023_fold_999_999/718/features_shared_model_perspective.csv")


def plot(load_csv, save_svg, odor_list, title, xoffset=0.11):
    odors = deepcopy(sorted(odor_list))
    
    odors.append('other')

    df = pd.read_csv(load_csv)

    df.loc[~df['odor'].isin(odors), 'odor'] = 'other'

    cmap = plt.get_cmap('hot')
    colors = cmap(np.linspace(0, 1, num=(len(odors)+1)))
    colors=colors[:len(odors)]

    markers = [None for _ in range(len(odors) * 2)]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for (odor, color, mark) in zip(odors, colors, markers):
        hex = mcolors.to_hex(color)
        subset = df.loc[df["odor"] == odor]

        alpha = 0.8
        s = 15

        if odor == 'other':
            alpha = 0.05
            hex = '#636363'
            mark = 'x'
        
        if mark:
            ax.scatter(x=subset['TSNE '+ str(0)] , y=subset['TSNE ' + str(1)], c=hex, label=odor, marker=mark, alpha=alpha, s=s)
        else:
            ax.scatter(x=subset['TSNE '+ str(0)] , y=subset['TSNE ' + str(1)], c=hex, label=odor, alpha=alpha, s=s)
        
        ax.set_xlabel('TSNE 0')
        ax.set_ylabel('TSNE 1')

    odor_labels = []
    for odor in odors:
        lb = odor
        if 'leffingwell' in odor:
            lb = odor.split('_')[0] + ' (Leffingwell)'
            
        elif 'ifra' in odor:
            lb = odor.split('_')[0] + ' (IFRA)'
        odor_labels.append(lb)
    
    plt.xticks([])
    plt.yticks([])

    plt.gcf().set_size_inches((5, 5))

    handles, _ = ax.get_legend_handles_labels()
    if len(odor_labels) < 4:
        handles.insert(1, mlines.Line2D([], [], color="none", label="", alpha=0))

    legend = fig.legend(handles=handles, labels=[h.get_label() for h in handles], title='odor', loc='upper left', bbox_to_anchor=(xoffset, 0.0), bbox_transform=plt.gcf().transFigure, ncols=3, fancybox=False, shadow=True)
    for l in legend.legend_handles:
        l.set_alpha(1.0)
    fig.align_ylabels() 
    fig.align_xlabels()

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(save_svg, format='pdf', dpi=1200, bbox_inches='tight')

leff_odor_list_bad = ['green', 'sweet']
leff_odor_list_good = ['sulfurous', 'nutty', 'spicy', 'ethereal', 'caramellic']

ifra_odor_list = ['fruity', 'pear', 'apple', 'pineapple']
shared_odor_list = ['woody', 'spicy', 'apple']

plot("../output/analysis/shared_model.csv", "../output/plots/fig8a.pdf", shared_odor_list, 'Separation of classes in shared model', xoffset=0.11)
plot("../output/analysis/leffingwell.csv", "../output/plots/fig8b.pdf", shared_odor_list, 'Separation of classes in Leffingwell head', xoffset=0.11)
plot("../output/analysis/ifra.csv", "../output/plots/fig8c.pdf", shared_odor_list, 'Separation of classes in IFRA head', xoffset=0.11)

plot("../output/analysis/leffingwell.csv", "../output/plots/fig9a.pdf", leff_odor_list_good, 'Clusters in Leffingwell head', xoffset=0.015)
plot("../output/analysis/ifra.csv", "../output/plots/fig9b.pdf", ifra_odor_list, 'Fruity cluster composition in IFRA head', xoffset=0.075)
plot("../output/analysis/leffingwell.csv", "../output/plots/fig9c.pdf", leff_odor_list_bad, 'Overlapping clusters in Leffingwell head', xoffset=0.11)

