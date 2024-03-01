import csv
import constants
import os
import json


def write_config(path, params):
    with open(path, 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)


def export_prediction_DREAM(path, predictions):
    with open(path, 'w') as output:
        writer = csv.writer(output, delimiter='\t')

        writer.writerow(['oID', 'descriptor', 'value', 'sigma'])

        for i in range(len(constants.DREAM_ODORS)):
            for cid, prediction in predictions.items():
                writer.writerow([str(cid), constants.DREAM_ODORS[i], str(prediction[i * 2]), str(prediction[(i * 2) + 1])])


def export_train_history(folder_path, loss_history, accuracy_history, loss_weight_history):
    with open(os.path.join(folder_path + 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f)

    with open(os.path.join(folder_path + 'accuracy_history.json'), 'w') as f:
        json.dump(accuracy_history, f)
    
    with open(os.path.join(folder_path + 'loss_balancing_history.json'), 'w') as f:
        json.dump(loss_weight_history, f)


def export_results(path, scores):
    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)
