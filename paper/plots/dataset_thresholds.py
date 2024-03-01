import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from itertools import chain
from collections import Counter

matplotlib.rcParams.update({'font.size': 12})

dataset_path = '../output/data/pyrfume/'

datasets = {}

dataset_names = os.listdir(dataset_path)
for dataset in dataset_names:
    db_name = dataset.replace(".json", "")

    if '_full' not in db_name:
        continue

    df = pd.read_json(dataset_path + db_name + '.json')
    
    datasets[db_name.replace('_full', '')] = df['Odors']

odor_counts = {}

for db_name, odors in datasets.items():
    flatten_list = list(chain(*odors))
    
    odor_counts[db_name] = Counter()

    for odor in flatten_list:
        odor_counts[db_name][db_name + '_' + odor] += 1
    
odor_counts['all'] = Counter()
for db_name in datasets.keys():
    odor_counts['all'] += odor_counts[db_name]

for db_name in odor_counts.keys():
    odor_counts[db_name] = sorted(odor_counts[db_name].items(), key=lambda x: x[1], reverse=True)

plot_data = {}

for db_name, counter in odor_counts.items():
    most_common = counter[0][1]

    plot_data[db_name] = {
        'dropped_x': [],
        'dropped_y': [],
        'remaining_x': [],
        'remaining_y': []
    }

    for threshold in range(101):
        num_dropped_classes = 0
        num_reamining_classes = len(counter)
        
        for x in counter:
            class_name, num_occurrences = x
            
            if threshold > num_occurrences:
                num_dropped_classes += 1
                num_reamining_classes -= 1

        
        plot_data[db_name]['dropped_x'].append(threshold)
        plot_data[db_name]['dropped_y'].append(num_dropped_classes)

        plot_data[db_name]['remaining_x'].append(threshold)
        plot_data[db_name]['remaining_y'].append(num_reamining_classes)

# Find intersection between dropped and remaining class amounts
x_points = plot_data['all']['dropped_x']

y_dropped = plot_data['all']['dropped_y']
y_remaining = plot_data['all']['remaining_y']

for x in x_points:
    if x == 0:
        continue

    if y_remaining[x - 1] > y_dropped[x - 1] and y_remaining[x] < y_dropped[x]:
        print(f'Lost 50% of classes at threshold {x}')

# How many classes do we lose if we use 10/20 as thresholds?
lost_ratio_at_10 = y_dropped[10] / (y_remaining[10] + y_dropped[10])
lost_ratio_at_20 = y_dropped[20] / (y_remaining[20] + y_dropped[20])

print(f'Lost {lost_ratio_at_10 * 100.0:.2f}% ({y_dropped[10]}) classes at threshold 10')
print(f'Lost {lost_ratio_at_20 * 100.0:.2f}% ({y_dropped[20]}) classes at threshold 20')

total_num_classes = y_remaining[0] + y_dropped[0]
print(f'Total num classes: {total_num_classes}')

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5), squeeze=False)

db_counter = 0

sorted_names = sorted(plot_data.keys(), key=lambda x: x.lower())

for db_name in sorted_names:
    data = plot_data[db_name]

    if db_name != 'all':
        continue

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

    dropped_x = data['dropped_x']
    dropped_y = data['dropped_y']

    remaining_x = data['remaining_x']
    remaining_y = data['remaining_y']

    labels = ['_' for _ in range(4)]

    if db_counter == 0:
        labels[0] = 'dropped classes'
        labels[1] = 'remaining classes'
    
    ax[db_counter][0].step(dropped_x, dropped_y, label=labels[0])
    ax[db_counter][0].step(remaining_x, remaining_y, label=labels[1])

    dropped_y_diff = np.diff(dropped_y)
    ax[db_counter][1].step(dropped_x[1:], dropped_y_diff, label=labels[2])
    remaining_y_diff = np.diff(remaining_y)
    ax[db_counter][1].step(remaining_x[1:], remaining_y_diff, label=labels[3])
    
    ax[db_counter][0].set_title(name_to_display)
    ax[db_counter][1].set_title(name_to_display + ' (first order difference)')
    ax[db_counter][0].set_xlabel('Class sample size threshold')
    ax[db_counter][1].set_xlabel('Class sample size threshold')

    db_counter += 1

legend = fig.legend(title='Number of', loc='upper center', fancybox=True, shadow=True, ncols=2, bbox_to_anchor=(0.5, 0.0))

fig.tight_layout()

plt.savefig('../output/plots/fig2.pdf', bbox_extra_artists=(legend,), bbox_inches='tight', dpi=1200)
plt.show()
