import pandas as pd
import numpy as np
import csv
from utility.score import DREAM_ODORS, calculate_scores_from_submission
from joblib import load
import sys



if sys.argv[1] == 'single':
    NUM_FOLDS = 1
    FOLD_IDS = [999]
    STUDY_NAME = "11082023_fold_999"
elif sys.argv[1] == 'ensemble':
    NUM_FOLDS = 10
    FOLD_IDS = np.arange(NUM_FOLDS)
    STUDY_NAME = "07082023"
else:
    raise Exception

DREAM_SCALER = load('../output/preprocessing/dream_scaler.joblib')

def _unscale_dream(predictions):
    unscaled_values = DREAM_SCALER.inverse_transform(np.array(predictions).reshape(1, -1))
    unscaled_values = unscaled_values.tolist()[0]

    return unscaled_values

ensemble_results = []

submission_header = ['#oID', 'descriptor', 'value', 'sigma']

best_models = {}

for fold_id in FOLD_IDS:
    fold_df = pd.read_json(f'../output/study_results/{STUDY_NAME}_{fold_id}.json', orient='index')
    
    fold_df['fold_id'] = fold_id
    fold_df['mse_loss'] = pd.json_normalize(fold_df['user_attrs'])['mse_loss']
    fold_df['z_score'] = pd.json_normalize(fold_df['user_attrs'])['z_score']
    
    fold_df = fold_df.sort_values(by=["z_score"], ascending=False)

    best_fold_model = fold_df.iloc[:1]
    best_fold_model['orig_idx'] = best_fold_model.index.values[:1]

    best_models[fold_id] = best_fold_model

    print(f'Fold #{fold_id}: N = {fold_df.shape[0]} (ID: {best_models[fold_id].index.values[0]})')


cid_order = []
predictions = {}

for fold_id in FOLD_IDS:
    for model_idx in best_models[fold_id].index:
        prediction = best_models[fold_id].loc[model_idx]['user_attrs']
        prediction = prediction['test_predictions']

        for cid in prediction.keys():
            unscaled_pred = _unscale_dream(prediction[cid])

            if cid not in cid_order:
                cid_order.append(cid)
            
            if cid not in predictions:
                predictions[cid] = {}
            
            for class_idx in range(42):
                odour_class = DREAM_ODORS[class_idx // 2]
                if odour_class not in predictions[cid]:
                    predictions[cid][odour_class] = {
                        'MEAN': [],
                        'STD': []
                    }
                
                if class_idx % 2 == 0:
                    predictions[cid][odour_class]['MEAN'].append(unscaled_pred[class_idx])
                else:
                    predictions[cid][odour_class]['STD'].append(unscaled_pred[class_idx])

# Calculate averaged predictions
new_predictions = {}
for cid in cid_order:
    new_predictions[cid] = {}

    for odour_class in DREAM_ODORS:
        avg_mean = np.average(predictions[cid][odour_class]['MEAN'])
        avg_std  = np.average(predictions[cid][odour_class]['STD'])

        new_predictions[cid][odour_class] = {
            'MEAN': avg_mean,
            'STD': avg_std
        }

# Write output
with open(f'../output/predictions/{sys.argv[1]}_DREAM_predictions.txt', 'w') as f:
    csv_writer = csv.writer(f, delimiter='\t')

    csv_writer.writerow(submission_header)

    for cid in cid_order:
        for odour_class in DREAM_ODORS:
            mean = new_predictions[cid][odour_class]['MEAN']
            std = new_predictions[cid][odour_class]['STD']

            csv_writer.writerow([cid, odour_class, mean, std])

scores = calculate_scores_from_submission('../data/dream/GSs2_new.txt', f'../output/predictions/{sys.argv[1]}_DREAM_predictions.txt')

print(f'DREAM test-set: {scores["z_score"]:.3f} (Z-Score) {scores["mse_loss"]:.3f} (MSE loss)')

best_configs = pd.concat(best_models.values(), ignore_index=True)
best_configs = best_configs.drop(columns=["user_attrs", "objective_value"])
best_configs.index.names = ["idx"]

output_path = f"../output/study_results/{STUDY_NAME}_best_models.csv"
best_configs.to_csv(output_path)

print(f"Saved best model parameters to {output_path}")
