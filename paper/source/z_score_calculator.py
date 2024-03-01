import numpy as np
import csv
import sys
from sklearn.metrics import mean_squared_error

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
def calculate_z_score(
    mean_intensity_pearson,
    mean_pleasantness_pearson,
    mean_odour_pearson,
    std_intensity_pearson,
    std_pleasantness_pearson,
    std_odour_pearson,
):

    sd = [0.119307474, 0.126471379, 0.026512975, 0.119494682, 0.11490659, 0.02808499]

    z_score = mean_intensity_pearson / sd[0]
    z_score += mean_pleasantness_pearson / sd[1]
    z_score += mean_odour_pearson / sd[2]
    z_score += std_intensity_pearson / sd[3]
    z_score += std_pleasantness_pearson / sd[4]
    z_score += std_odour_pearson / sd[5]

    z_score = z_score / 6.0

    return z_score


def __read_DREAM_submission(filename):
    cid_order = []
    data = {}

    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter="\t")

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

            if not data.get(cid, ""):
                data[cid] = {}

            data[cid][type] = (mean, std)

    return cid_order, data


def calculate_z_score_from_submission(
    ground_truth_file, submission_file, verbose=False, return_mse_loss=False
):
    cid_order, ground_truth = __read_DREAM_submission(ground_truth_file)
    _, predicted_values = __read_DREAM_submission(submission_file)

    ground_truth_mean_intensity = []
    ground_truth_mean_pleasantness = []
    ground_truth_mean_odours = {}
    ground_truth_std_intensity = []
    ground_truth_std_pleasantness = []
    ground_truth_std_odours = {}

    predicted_mean_intensity = []
    predicted_mean_pleasantness = []
    predicted_mean_odours = {}
    predicted_std_intensity = []
    predicted_std_pleasantness = []
    predicted_std_odours = {}

    for ordered_cid in cid_order:
        for cid, data in predicted_values.items():
            if cid == ordered_cid:
                for k, v in data.items():
                    if k == "INTENSITY/STRENGTH":
                        predicted_mean_intensity.append(v[0])
                        predicted_std_intensity.append(v[1])
                    elif k == "VALENCE/PLEASANTNESS":
                        predicted_mean_pleasantness.append(v[0])
                        predicted_std_pleasantness.append(v[1])
                    else:
                        if not predicted_mean_odours.get(k, ""):
                            predicted_mean_odours[k] = []
                            predicted_std_odours[k] = []

                        predicted_mean_odours[k].append(v[0])
                        predicted_std_odours[k].append(v[1])
                break

        for cid, data in ground_truth.items():
            if cid == ordered_cid:
                for k, v in data.items():
                    if k == "INTENSITY/STRENGTH":
                        ground_truth_mean_intensity.append(v[0])
                        ground_truth_std_intensity.append(v[1])
                    elif k == "VALENCE/PLEASANTNESS":
                        ground_truth_mean_pleasantness.append(v[0])
                        ground_truth_std_pleasantness.append(v[1])
                    else:
                        if not ground_truth_mean_odours.get(k, ""):
                            ground_truth_mean_odours[k] = []
                            ground_truth_std_odours[k] = []

                        ground_truth_mean_odours[k].append(v[0])
                        ground_truth_std_odours[k].append(v[1])
                break

    pearson_mean_intensity = calculate_pearson(
        ground_truth_mean_intensity, predicted_mean_intensity
    )
    pearson_mean_pleasantness = calculate_pearson(
        ground_truth_mean_pleasantness, predicted_mean_pleasantness
    )

    pearson_mean_odours = 0.0
    pearson_std_odours = 0.0

    for odour in predicted_mean_odours.keys():
        pearson_mean_odours += calculate_pearson(
            ground_truth_mean_odours[odour], predicted_mean_odours[odour]
        )
        pearson_std_odours += calculate_pearson(
            ground_truth_std_odours[odour], predicted_std_odours[odour]
        )

    pearson_mean_odours /= 19.0
    pearson_std_odours /= 19.0

    pearson_std_intensity = calculate_pearson(
        ground_truth_std_intensity, predicted_std_intensity
    )
    pearson_std_pleasantness = calculate_pearson(
        ground_truth_std_pleasantness, predicted_std_pleasantness
    )

    if verbose:
        print(f"mean_odors: {pearson_mean_odours}")
        print(f"mean_intensity: {pearson_mean_intensity}")
        print(f"mean_pleasantness: {pearson_mean_pleasantness}")
        print(f"std_odors: {pearson_std_odours}")
        print(f"std_intensity: {pearson_std_intensity}")
        print(f"std_pleasantness: {pearson_std_pleasantness}")

    # Flatten ground truth and predictions to lists
    flattened_ground_truth = (
        ground_truth_mean_intensity
        + ground_truth_std_intensity
        + ground_truth_mean_pleasantness
        + ground_truth_std_pleasantness
    )
    flattened_prediction = (
        predicted_mean_intensity
        + predicted_std_intensity
        + predicted_mean_pleasantness
        + predicted_std_pleasantness
    )
    for odour in predicted_mean_odours.keys():
        flattened_ground_truth.extend(ground_truth_mean_odours[odour])
        flattened_ground_truth.extend(ground_truth_std_odours[odour])

        flattened_prediction.extend(predicted_mean_odours[odour])
        flattened_prediction.extend(predicted_std_odours[odour])

    z_score = calculate_z_score(
        pearson_mean_intensity,
        pearson_mean_pleasantness,
        pearson_mean_odours,
        pearson_std_intensity,
        pearson_std_pleasantness,
        pearson_std_odours,
    )

    if return_mse_loss:
        return z_score, mean_squared_error(flattened_ground_truth, flattened_prediction)
    else:
        return z_score


if __name__ == "__main__":
    ground_truth_filename = sys.argv[1]
    submission_filename = sys.argv[2]
    z_score = calculate_z_score_from_submission(
        ground_truth_filename, submission_filename, verbose=False#True
    )

    print(f" Z-Score: {z_score}")
