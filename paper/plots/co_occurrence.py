import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

matplotlib.rcParams.update({'font.size': 12})

df = pd.read_json('../output/preprocessing/known_combinations.json', orient='index')
classes_df = pd.read_json('../output/preprocessing/pyrfume_classes.json', orient='index')

db_classes = {}

for db_name in ['arctander', 'ifra_2019', 'leffingwell', 'sigma_2014']:
    db_classes[db_name] = sorted(classes_df.loc[db_name].dropna().values.tolist())

z_values = {}

for db_name in db_classes.keys():
    num_classes = len(db_classes[db_name])

    z_values[db_name] = np.zeros((num_classes, num_classes), dtype=np.int32)

    db_df = df.loc[db_name].dropna()

    for x_class_name in db_df.index:
        x_index = db_classes[db_name].index(x_class_name)

        cooccurrences = db_df.loc[x_class_name]
        for y_class_name in cooccurrences:
            y_index = db_classes[db_name].index(y_class_name)

            z_values[db_name][x_index][y_index] += 1

fig, axs = plt.subplots(1, 4, figsize=(12, 6))

x = 0

for db_name in z_values.keys():
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

    im = axs[x].imshow(z_values[db_name])

    axs[x].set_xticks([])
    axs[x].set_yticks([])
    axs[x].set_title(name_to_display)

    x += 1

patches = [mpatches.Patch(color=im.cmap(im.norm(0)), label='exists'), mpatches.Patch(color=im.cmap(im.norm(1)), label='does not exist'),]
legend = fig.legend(handles=patches, title='Pair combination', loc='lower center', bbox_to_anchor=(0.5, 0.125), fancybox=False, shadow=True, ncols=2)

fig.tight_layout()
plt.savefig('../output/plots/fig4.pdf', dpi=1200, bbox_extra_artists=(legend,), bbox_inches='tight')
plt.show()