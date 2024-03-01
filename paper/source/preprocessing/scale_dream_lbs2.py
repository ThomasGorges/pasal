import pandas as pd
import numpy as np
from joblib import load
import csv


dream_scaler = load('../../output/preprocessing/dream_scaler.joblib')

fusion_df = pd.read_json('../../output/preprocessing/fusion.json', orient='index')
fusion_df.index.names = ['CID']

dream_val_df = pd.read_csv('../../data/dream/LBs2.txt', delimiter='\t')
dream_val_df = dream_val_df.set_index('#oID')
dream_val_cids = dream_val_df.index.values

dream_class_names = pd.read_csv('../../output/data/dream/TrainSetProcessedAvg.csv').set_index('CID').columns.values.tolist()

cleaned_class_names = []
for class_idx in range(0, len(dream_class_names), 2):
    class_name = dream_class_names[class_idx]
    class_name = class_name[class_name.rfind("_") + 1:]

    if class_name == "CHEMICAL":
        class_name = " CHEMICAL"
    
    cleaned_class_names.append(class_name)

new_val_labels = {}
for cid in dream_val_cids:
    mol = fusion_df.loc[cid]

    dream_mean = mol['dream_mean']
    dream_std = mol['dream_std']
    
    # Reassemble annotations in correct order
    labels = []
    for class_name in cleaned_class_names:
        file_entry = dream_val_df.loc[(dream_val_df.index == cid) & (dream_val_df['descriptor'] == class_name)]
        mean_value = file_entry['value'].tolist()[0]
        std_value = file_entry['sigma'].tolist()[0]

        labels.append(mean_value)
        labels.append(std_value)
    
    scaled_values = dream_scaler.transform(np.array(labels).reshape(1, -1))[0]
    new_val_labels[cid] = scaled_values


with open('../../output/data/dream/LBs2_scaled.csv', 'w') as f:
    csv_writer = csv.writer(f, delimiter='\t')
    csv_writer.writerow(['#oID', 'descriptor', 'value', 'sigma'])

    for class_idx in range(len(cleaned_class_names)):
        class_name = cleaned_class_names[class_idx]

        for cid in dream_val_cids:
            mean_value = new_val_labels[cid][class_idx * 2 + 0]
            std_value = new_val_labels[cid][class_idx * 2 + 1]

            row = [str(cid), str(class_name), mean_value, std_value]
            csv_writer.writerow(row)
