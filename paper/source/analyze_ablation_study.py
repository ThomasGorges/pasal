import pandas as pd
import numpy as np
from joblib import load
import constants
from utility.score import calculate_scores_from_submission_in_memory

NUM_DATASETS = 5

DREAM_SCALER = load("../output/preprocessing/dream_scaler.joblib")
LBS2_CID_ORDER, LBS2_GROUND_TRUTH = constants.read_DREAM_submission('../data/dream/LBs2.txt')

def _unscale_dream(predictions):
    result = {}

    for cid in predictions.keys():
        unscaled_values = DREAM_SCALER.inverse_transform(
            np.array(predictions[cid]).reshape(1, -1)
        )
        result[cid] = unscaled_values.tolist()[0]

    return result


for i in range(NUM_DATASETS):
    df = pd.read_json(f"../output/study_results/ablation_{i + 1}_dataset_seed_0_999.json", orient="index")

    z_scores = []
    mse_losses = []

    for run_id in df.index:
        run_attribs = df['user_attrs'][run_id]

        predictions = run_attribs['val_predictions']
        predictions = _unscale_dream(predictions)

        scores = calculate_scores_from_submission_in_memory(
                    LBS2_CID_ORDER,
                    LBS2_GROUND_TRUTH,
                    predictions,
        )

        z_scores.append(scores['z_score'])
        mse_losses.append(scores['mse_loss'])

    avg_z_score = f"{np.average(z_scores):.2f}"
    std_z_score = f"{np.std(z_scores):.2f}"

    avg_mse = f"{np.average(mse_losses):.2f}"
    std_mse = f"{np.std(mse_losses):.2f}"

    print(f"#{i + 1} datasets:\n\tMSE    : {avg_mse} \u00B1 {std_mse}\n\tZ-Score:  {avg_z_score} \u00B1 {std_z_score}")
