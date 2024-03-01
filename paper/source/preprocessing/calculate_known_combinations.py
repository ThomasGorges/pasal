import pandas as pd
import json


classes_df = pd.read_json(
    "../../output/preprocessing/pyrfume_classes.json", orient="index"
)

datasets = classes_df.index.values


dataset_classes = {}

print("Found following datasets:")
for dataset in datasets:
    print(f"\t- {dataset}")

    dataset_classes[dataset] = classes_df.loc[dataset].dropna().tolist()


cooccurence_matrices = {}
known_combinations = {}

for dataset in datasets:
    cooccurence_matrices[dataset] = []
    known_combinations[dataset] = {}

    class_list = classes_df.loc[dataset].dropna().values

    for _ in class_list:
        cooccurence_matrices[dataset].append([0] * len(class_list))

molecules = pd.read_json("../../output/preprocessing/fusion.json", orient="index")

train_folds = pd.read_json("../../output/preprocessing/train_folds.json", orient="index")
train_cids = train_folds.loc[999]['train_cids']

molecules = molecules.loc[train_cids]

for cid in molecules.index:
    molecule = molecules.loc[cid]
    molecule = molecule.drop(["dream_mean", "dream_std", "IsomericSMILES"]).dropna()

    mol_datasets = molecule.index.values

    for dataset in mol_datasets:
        odours = molecule.loc[dataset]
        odour_ids = [dataset_classes[dataset].index(x) for x in odours]

        for odour_id in odour_ids:
            for odour_id2 in odour_ids:
                if odour_id == odour_id2:
                    continue

                cooccurence_matrices[dataset][odour_id][odour_id2] += 1

                odour_name = dataset_classes[dataset][odour_id]
                odour_name2 = dataset_classes[dataset][odour_id2]

                # Do we already know this combination? -> skip
                is_combo_known = False

                if odour_name not in known_combinations[dataset]:
                    known_combinations[dataset][odour_name] = []

                if odour_name2 not in known_combinations[dataset]:
                    known_combinations[dataset][odour_name2] = []

                if (
                    odour_name in known_combinations[dataset][odour_name2]
                    or odour_name2 in known_combinations[dataset][odour_name]
                ):
                    is_combo_known = True

                if not is_combo_known:
                    known_combinations[dataset][odour_name].append(odour_name2)
                    known_combinations[dataset][odour_name2].append(odour_name)


# Export data
with open(f"../../output/preprocessing/known_combinations.json", "w") as f:
    json.dump(known_combinations, f, indent=4, sort_keys=True)
