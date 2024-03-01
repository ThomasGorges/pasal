import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm, shapiro, levene, ttest_ind

matplotlib.rc('font', size=13)

ALPHA = 0.01

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

ground_truth_df = pd.read_csv('../data/dream/GSs2_newProcessed.csv')
ground_truth_df = ground_truth_df.set_index('CID')
ground_truth_df = ground_truth_df[[col for col in ground_truth_df.columns if 'MEAN_' in col]]
ground_truth_df.columns = ground_truth_df.columns.map(lambda x: x[5:])


df = pd.read_csv('../data/dream/TestSet.txt', delimiter='\t')
df = df.rename(columns={'Compound Identifier': 'CID'})
df = df.set_index('CID')
df = df.drop(columns=['Odor', 'Replicate', 'Intensity', 'Dilution'])

cids = np.unique(ground_truth_df.index)

approach_models = [
    '07082023_0/625',
    '07082023_1/270',
    '07082023_2/958',
    '07082023_3/672',
    '07082023_4/903',
    '07082023_5/716',
    '07082023_6/227',
    '07082023_7/964',
    '07082023_8/401',
    '07082023_9/626',
    '11082023_fold_999_999/718',
]

for model_idx, model in enumerate(approach_models):
    approach_df = pd.read_csv(f'../output/study_results/{model}/prediction_GSs2.txt', delimiter='\t')
    approach_df = approach_df.rename(columns={'oID': 'CID'})
    approach_df = approach_df.drop(columns=['sigma'])
    approach_df = approach_df.set_index('CID')

    predictions = {}

    for cid in cids:
        predictions[cid] = [model_idx + 50, *approach_df.loc[cid]['value'].values]

    prediction_df = pd.DataFrame.from_dict(predictions).T
    prediction_df.index = cids
    prediction_df.columns = ['subject #', *ground_truth_df.columns]

    df = pd.concat([df, prediction_df])

subject_ids = (np.arange(49) + 1).tolist()
subject_ids.extend((np.arange(len(approach_models)) + 50).tolist())

ground_truth_int = ground_truth_df[['INTENSITY/STRENGTH']]
ground_truth_val = ground_truth_df[['VALENCE/PLEASANTNESS']]
ground_truth_odors = ground_truth_df.drop(columns=['INTENSITY/STRENGTH', 'VALENCE/PLEASANTNESS'])


print(f'alpha = {ALPHA}')

for name, file_suffix, cols in [
        ('intensity', 'b', ground_truth_int),
        ('pleasantness', 'c', ground_truth_val),
        ('odors', 'a', ground_truth_odors)
    ]:
    print(f'----- {name} -----')

    pearson_corrs = {}

    for subject_id in subject_ids:
        subject_df = df[df['subject #'] == subject_id]
        subject_df = subject_df[ground_truth_df.columns]

        subject_pearson = []

        for col in cols:
            pred = subject_df[col].dropna()
            y = ground_truth_df[col]

            y = y.loc[pred.index]
            y = y.reindex(pred.index)

            subject_pearson.append(np.abs(calculate_pearson(y.values, pred.values)))

        pearson_corrs[subject_id] = np.average(subject_pearson)

    fig, ax = plt.subplots(figsize=(5, 5))

    num_bins = 50
    bins = np.linspace(0.0, 1.0, num_bins)
    
    human_dist = list(pearson_corrs.values())[:-len(approach_models)]
    human_mean, human_sigma = norm.fit(human_dist)

    ai_dist = list(pearson_corrs.values())[-len(approach_models):]
    ai_mean, ai_sigma = norm.fit(ai_dist)

    plt.hist([human_dist, ai_dist], bins=(np.arange(num_bins) / num_bins) - (1.0 / num_bins) / 2, histtype='bar', color=['lightskyblue', 'orange'], label=['Human', 'Our approach'])

    plt.plot(np.arange(num_bins) * (1.0 / num_bins), norm.pdf(np.arange(num_bins) * (1.0 / num_bins), human_mean, human_sigma), '--', path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], color='lightskyblue', label='_Human')
    plt.plot(np.arange(num_bins) * (1.0 / num_bins), norm.pdf(np.arange(num_bins) * (1.0 / num_bins), ai_mean, ai_sigma), '--', path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()], color='orange', label='_Our approach')

    # Test if data is normal distributed
    human_shapiro = shapiro(human_dist)
    ai_shapiro = shapiro(ai_dist)

    print(f'Shapiro-Wilk-test: H0 = data normal distributed')
    if human_shapiro.pvalue < ALPHA:
        print(f'Shapiro-Wilk-test (Human): H0 rejected - {human_shapiro}')
    else:
        print(f'Shapiro-Wilk-test (Human): H0 not rejected - {human_shapiro}')

    if ai_shapiro.pvalue < ALPHA:
        print(f'Shapiro-Wilk-test (AI): H0 rejeceted - {ai_shapiro}')
    else:
        print(f'Shapiro-Wilk-test (AI): H0 not rejected - {ai_shapiro}')

    # We assume under the central limit theorem, that the mean tends towards a normal distribution

    # Check if variance of the two groups are equal
    levene_res = levene(human_dist, ai_dist)
    
    print('Levene-test: H0 = equal variances')
    if levene_res.pvalue < ALPHA:
        print(f'Levene-test: H0 rejected - {levene_res}')
    else:
        print(f'Levene-test: H0 not rejected - {levene_res}')
    
    welch_res = ttest_ind(human_dist, ai_dist, equal_var=False, random_state=0)
    print('Welch\'s t-test: H0 = equal means')
    if welch_res.pvalue < ALPHA:
        print(f'Welch\'s t-test: H0 rejected - {welch_res}')
    else:
        print(f'Welch\'s t-test: H0 not rejected - {welch_res}')

    ax.set_xlim((0.0, 1.0))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r'Absolute pearson correlation$\uparrow$')
    ax.set_ylabel('Frequency / Density')

    #ax.set_title(f'Performance comparison ({name})')

    ax.legend()

    fig.tight_layout()

    plt.savefig(f'../output/plots/fig11{file_suffix}.pdf')
