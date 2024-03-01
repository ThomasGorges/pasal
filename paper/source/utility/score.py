import numpy as np
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error


DREAM_ODORS = ['INTENSITY/STRENGTH', 'VALENCE/PLEASANTNESS', 'BAKERY', 'SWEET', 'FRUIT', 'FISH', 'GARLIC', 'SPICES',
               'COLD', 'SOUR', 'BURNT', 'ACID', 'WARM', 'MUSKY', 'SWEATY', 'AMMONIA/URINOUS', 'DECAYED', 'WOOD',
               'GRASS', 'FLOWER', 'CHEMICAL']

# DREAM_Olfaction_scoring_Q2.pl is from https://github.com/Sage-Bionetworks-Challenges/OlfactionDREAMChallenge/blob/90adc4695cae6adb0e40222d21e2619b5b776ea0/src/main/resources/DREAM_Olfaction_scoring_Q2.pl

# Adapted from DREAM_Olfaction_scoring_Q2.pl
def __diff_sum(v):
    avg = np.average(v)
    sum = 0.0

    for i in range(len(v)):
        dif = v[i] - avg
        sum += dif * dif

    return np.sqrt(sum)


# Adapted from DREAM_Olfaction_scoring_Q2.pl
def calculate_pearson(ground_truth, predicted):
    diff_sum0 = __diff_sum(ground_truth)
    diff_sum1 = __diff_sum(predicted)
    avg0 = np.average(ground_truth)
    avg1 = np.average(predicted)

    sum = 0.0
    for i in range(len(ground_truth)):
        dif0 = ground_truth[i] - avg0
        dif1 = predicted[i] - avg1

        sum += dif0 * dif1

    den = diff_sum0 * diff_sum1
    pearson = 0.0
    if den != 0:
        pearson = sum / (diff_sum0 * diff_sum1)

    return pearson


# Adapted from DREAM_Olfaction_scoring_Q2.pl
def calculate_z_score(mean_intensity_pearson, mean_pleasantness_pearson, mean_odour_pearson,
                      std_intensity_pearson, std_pleasantness_pearson, std_odour_pearson):

    sd = [0.119307474, 0.126471379, 0.026512975, 0.119494682, 0.11490659, 0.02808499]

    z_score = mean_intensity_pearson / sd[0]
    z_score += mean_pleasantness_pearson / sd[1]
    z_score += mean_odour_pearson / sd[2]
    z_score += std_intensity_pearson / sd[3]
    z_score += std_pleasantness_pearson / sd[4]
    z_score += std_odour_pearson / sd[5]

    z_score = z_score / 6.0

    return z_score


def read_DREAM_submission(filename):
    cid_order = []
    data = {}

    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter='\t')

        header = None

        for row in csv_reader:
            if not header:
                header = row
                continue

            cid = row[0]
            type = row[1].strip()
            mean = float(row[2])
            std = float(row[3])

            if cid not in cid_order:
                cid_order.append(cid)

            if not data.get(cid, ''):
                data[cid] = {}

            data[cid][type] = (mean, std)

    return cid_order, data


def calculate_scores_from_submission(submission_file, ground_truth_filename):
    cid_order, ground_truth = read_DREAM_submission(ground_truth_filename)
    _, predicted_values = read_DREAM_submission(submission_file)

    ground_truth_mean_intensity = []
    ground_truth_mean_pleasantness = []
    ground_truth_mean_odors = {}
    ground_truth_std_intensity = []
    ground_truth_std_pleasantness = []
    ground_truth_std_odors = {}

    predicted_mean_intensity = []
    predicted_mean_pleasantness = []
    predicted_mean_odors = {}
    predicted_std_intensity = []
    predicted_std_pleasantness = []
    predicted_std_odors = {}

    for ordered_cid in cid_order:
        for cid, data in predicted_values.items():
            if cid == ordered_cid:
                for k, v in data.items():
                    if k == 'INTENSITY/STRENGTH':
                        predicted_mean_intensity.append(v[0])
                        predicted_std_intensity.append(v[1])
                    elif k == 'VALENCE/PLEASANTNESS':
                        predicted_mean_pleasantness.append(v[0])
                        predicted_std_pleasantness.append(v[1])
                    else:
                        if not predicted_mean_odors.get(k, ''):
                            predicted_mean_odors[k] = []
                            predicted_std_odors[k] = []

                        predicted_mean_odors[k].append(v[0])
                        predicted_std_odors[k].append(v[1])
                break

        for cid, data in ground_truth.items():
            if cid == ordered_cid:
                for k, v in data.items():
                    if k == 'INTENSITY/STRENGTH':
                        ground_truth_mean_intensity.append(v[0])
                        ground_truth_std_intensity.append(v[1])
                    elif k == 'VALENCE/PLEASANTNESS':
                        ground_truth_mean_pleasantness.append(v[0])
                        ground_truth_std_pleasantness.append(v[1])
                    else:
                        if not ground_truth_mean_odors.get(k, ''):
                            ground_truth_mean_odors[k] = []
                            ground_truth_std_odors[k] = []

                        ground_truth_mean_odors[k].append(v[0])
                        ground_truth_std_odors[k].append(v[1])
                break

    pearson_mean_intensity = calculate_pearson(ground_truth_mean_intensity, predicted_mean_intensity)
    pearson_mean_pleasantness = calculate_pearson(ground_truth_mean_pleasantness, predicted_mean_pleasantness)

    pearson_mean_odors = 0.0
    pearson_std_odors = 0.0

    for odour in predicted_mean_odors.keys():
        pearson_mean_odors += calculate_pearson(ground_truth_mean_odors[odour], predicted_mean_odors[odour])
        pearson_std_odors += calculate_pearson(ground_truth_std_odors[odour], predicted_std_odors[odour])

    pearson_mean_odors /= 19.0
    pearson_std_odors /= 19.0

    pearson_std_intensity = calculate_pearson(ground_truth_std_intensity, predicted_std_intensity)
    pearson_std_pleasantness = calculate_pearson(ground_truth_std_pleasantness, predicted_std_pleasantness)

    z_score = calculate_z_score(pearson_mean_intensity,
                                pearson_mean_pleasantness,
                                pearson_mean_odors,
                                pearson_std_intensity,
                                pearson_std_pleasantness,
                                pearson_std_odors)

    # Flatten ground truth and predictions to lists
    flattened_ground_truth = ground_truth_mean_intensity + ground_truth_std_intensity + ground_truth_mean_pleasantness + ground_truth_std_pleasantness
    flattened_prediction = predicted_mean_intensity + predicted_std_intensity + predicted_mean_pleasantness + predicted_std_pleasantness
    for odour in predicted_mean_odors.keys():
        flattened_ground_truth.extend(ground_truth_mean_odors[odour])
        flattened_ground_truth.extend(ground_truth_std_odors[odour])

        flattened_prediction.extend(predicted_mean_odors[odour])
        flattened_prediction.extend(predicted_std_odors[odour])
    
    scores = {
        'z_score': z_score,
        'pearson_mean_intensity': pearson_mean_intensity,
        'pearson_mean_pleasantness': pearson_mean_pleasantness,
        'pearson_mean_odors': pearson_mean_odors,
        'pearson_std_intensity': pearson_std_intensity,
        'pearson_std_pleasantness': pearson_std_pleasantness,
        'pearson_std_odors': pearson_std_odors,
        'mae_loss': mean_absolute_error(flattened_ground_truth, flattened_prediction),
        'mse_loss': mean_squared_error(flattened_ground_truth, flattened_prediction)
    }

    return scores

def calculate_scores_from_submission_in_memory(cid_order, ground_truth, predictions):
    predicted_values = {}

    for i in range(len(DREAM_ODORS)):
        for cid, prediction in predictions.items():
            cid = str(cid)
            type = DREAM_ODORS[i].strip()
            mean = float(prediction[i * 2])
            std = float(prediction[(i * 2) + 1])

            if not predicted_values.get(cid, ''):
                predicted_values[cid] = {}

            predicted_values[cid][type] = (mean, std)
    
    ground_truth_mean_intensity = []
    ground_truth_mean_pleasantness = []
    ground_truth_mean_odors = {}
    ground_truth_std_intensity = []
    ground_truth_std_pleasantness = []
    ground_truth_std_odors = {}

    predicted_mean_intensity = []
    predicted_mean_pleasantness = []
    predicted_mean_odors = {}
    predicted_std_intensity = []
    predicted_std_pleasantness = []
    predicted_std_odors = {}

    for ordered_cid in cid_order:
        for cid, data in predicted_values.items():
            if cid == ordered_cid:
                for k, v in data.items():
                    if k == 'INTENSITY/STRENGTH':
                        predicted_mean_intensity.append(v[0])
                        predicted_std_intensity.append(v[1])
                    elif k == 'VALENCE/PLEASANTNESS':
                        predicted_mean_pleasantness.append(v[0])
                        predicted_std_pleasantness.append(v[1])
                    else:
                        if not predicted_mean_odors.get(k, ''):
                            predicted_mean_odors[k] = []
                            predicted_std_odors[k] = []

                        predicted_mean_odors[k].append(v[0])
                        predicted_std_odors[k].append(v[1])
                break

        for cid, data in ground_truth.items():
            if cid == ordered_cid:
                for k, v in data.items():
                    if k == 'INTENSITY/STRENGTH':
                        ground_truth_mean_intensity.append(v[0])
                        ground_truth_std_intensity.append(v[1])
                    elif k == 'VALENCE/PLEASANTNESS':
                        ground_truth_mean_pleasantness.append(v[0])
                        ground_truth_std_pleasantness.append(v[1])
                    else:
                        if not ground_truth_mean_odors.get(k, ''):
                            ground_truth_mean_odors[k] = []
                            ground_truth_std_odors[k] = []

                        ground_truth_mean_odors[k].append(v[0])
                        ground_truth_std_odors[k].append(v[1])
                break

    pearson_mean_intensity = calculate_pearson(ground_truth_mean_intensity, predicted_mean_intensity)
    pearson_mean_pleasantness = calculate_pearson(ground_truth_mean_pleasantness, predicted_mean_pleasantness)

    pearson_mean_odors = 0.0
    pearson_std_odors = 0.0

    for odour in predicted_mean_odors.keys():
        pearson_mean_odors += calculate_pearson(ground_truth_mean_odors[odour], predicted_mean_odors[odour])
        pearson_std_odors += calculate_pearson(ground_truth_std_odors[odour], predicted_std_odors[odour])

    pearson_mean_odors /= 19.0
    pearson_std_odors /= 19.0

    pearson_std_intensity = calculate_pearson(ground_truth_std_intensity, predicted_std_intensity)
    pearson_std_pleasantness = calculate_pearson(ground_truth_std_pleasantness, predicted_std_pleasantness)

    z_score = calculate_z_score(pearson_mean_intensity,
                                pearson_mean_pleasantness,
                                pearson_mean_odors,
                                pearson_std_intensity,
                                pearson_std_pleasantness,
                                pearson_std_odors)

    # Flatten ground truth and predictions to lists
    flattened_ground_truth = ground_truth_mean_intensity + ground_truth_std_intensity + ground_truth_mean_pleasantness + ground_truth_std_pleasantness
    flattened_prediction = predicted_mean_intensity + predicted_std_intensity + predicted_mean_pleasantness + predicted_std_pleasantness
    for odour in predicted_mean_odors.keys():
        flattened_ground_truth.extend(ground_truth_mean_odors[odour])
        flattened_ground_truth.extend(ground_truth_std_odors[odour])

        flattened_prediction.extend(predicted_mean_odors[odour])
        flattened_prediction.extend(predicted_std_odors[odour])

    scores = {
        'z_score': z_score,
        'pearson_mean_intensity': pearson_mean_intensity,
        'pearson_mean_pleasantness': pearson_mean_pleasantness,
        'pearson_mean_odors': pearson_mean_odors,
        'pearson_std_intensity': pearson_std_intensity,
        'pearson_std_pleasantness': pearson_std_pleasantness,
        'pearson_std_odors': pearson_std_odors,
        'mae_loss': mean_absolute_error(flattened_ground_truth, flattened_prediction),
        'mse_loss': mean_squared_error(flattened_ground_truth, flattened_prediction)
    }

    return scores


def calculate_folded_scores_in_memory(ground_truth, predictions):
    flattened_ground_truth = []
    flattened_prediction = []

    for cid in ground_truth.keys():
        flattened_ground_truth.extend(ground_truth[cid])
        flattened_prediction.extend(predictions[cid])

    scores = {
        "mae_loss": mean_absolute_error(flattened_ground_truth, flattened_prediction),
        "mse_loss": mean_squared_error(flattened_ground_truth, flattened_prediction),
    }

    return scores

def denormalize_scores(scores):
    with open('data/dream/normalization_values.txt') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        min, max = lines[i].strip().split(',')

        scores[i] = ((scores[i] / 100.0) * float(max)) + float(min)

    return scores

