# PASAL
This repository contains the implementation of the paper "PASAL: Progress- and sparsity-aware loss balancing for heterogeneous dataset fusion" from Gorges et al.

This is a fork of the initial implementation hosted on the [Fraunhofer GitLab](https://gitlab.cc-asp.fraunhofer.de/pasal/pasal). This fork includes changes required for the revision of the article. The last commit from Fraunhofer was commit 6308136f with the message "Replace files with their compressed versions". Subsequent commits were made by the Friedrich-Alexander-Universität Erlangen-Nürnberg.

### Getting started
A Dockerfile is provided to ease with the setup of the environment.

To build the image:
```
docker build . -t pasal
```

To create the container:
```
docker run -it -v ./paper:/home/pasal/paper:Z pasal
```

Some files are compressed and need to be decompressed. This can be achieved, for example, with:
```
find . -type f -name '*.tar.xz' -execdir tar -xJf {} \;
find . -type f -name '*.xz' -print0 | xargs -0 xz -d
```

## Preprocessing
Preprocessing is split into multiple parts:

### Dataset preprocessing

To process the DREAM Olfaction Prediction Challenge datasets and datasets from Pyrfume, run:
```
cd paper/source/preprocessing/datasets/
./preprocess.sh
```

### Old feature calculation
During the development, the input for the feature selection was frozen to ensure reproducibility. These files are located at source/preprocessing/old_features/.

Steps to reproduce:
```
cd paper/source/preprocessing/old_features/datasets/
./preprocess.sh
cd ..
python3 fusion.py
python3 to_sdf.py
python3 calculate_map4.py
python3 calculate_mordred_features_.py
```

PaDEL features were calculated with the GUI, which can be obtained [here](http://www.yapcwsoft.com/dd/padeldescriptor/).

Note that the feature calculation relies on third-party tools, that can produce non-deterministic results.

### Feature calculation
To run the feature calculation steps, execute following command, which may take a while:
```
cd paper/source/preprocessing/
./preprocess.sh
```

## Training

The flag "num_worker" must be adjusted based on the existing hardware setup.

Ablation study:
```
cd paper/source/
./run_ablation_study.sh
```

Hyperparameter search (single):
```
cd paper/source/
python3 main.py --study_name 11082023_fold_999 --num_workers 70 --num_models 1000 --random_search_seed 0 --fold_id 999
python3 main.py --study_name 11082023_fold_999 --fetch_results --fold_id 999
```

Hyperparameter search (ensemble):
```
cd paper/source/
python3 main.py --study_name 07082023 --num_workers 70 --num_models 1000 --random_search_seed 0
python3 main.py --study_name 07082023 --fetch_results
```

Output will be saved at paper/output/study_results/.

## Results

### Single model (non ensemble)
```
cd paper/source/
python3 analyze_results.py single
```

### Ensemble
```
cd paper/source/
python3 analyze_results.py ensemble
```

Respective Z-Score will be printed and the predictions will be saved at paper/output/predictions/.

### Retraining models & loss balancing

For constant loss balancing:
```
python3 main.py --retrain study:///11082023_fold_999_999/718 --fold_id 999 --alpha 0.0 --beta 0.0
```

Alpha and beta can be evalauted with different values. Evaluated combinations for alpha are 1.4, 1.5 and 1.6 & for beta 0.7, 0.8 and 0.9.

## Plots
To reproduce the plots, execute following commands:

```
cd paper/plots/
python3 [FILE_NAME]
```

Output will be saved at paper/output/plots/. Significance test is included in the human_performance.py script.

## Third-party tools & data
This software uses third-party sources. See the license folder.

### Data
Following additional third-party data is used:
- DREAM Olfaction Prediction Challenge: https://www.synapse.org/#!Synapse:syn2811262/wiki/78375 & https://github.com/dream-olfaction/olfaction-prediction
- arctander_1960: https://github.com/pyrfume/pyrfume-data/tree/main/arctander_1960
- leffingwell: https://github.com/pyrfume/pyrfume-data/tree/main/leffingwell
- sigma_2014: https://github.com/pyrfume/pyrfume-data/tree/main/sigma_2014
- ifra_2019: https://github.com/pyrfume/pyrfume-data/tree/main/ifra_2019 & https://ifrafragrance.org/priorities/ingredients/glossary

This work uses information derived from the IFRA Fragrance Ingredient Glossary, developed by The International Fragrance Association.

## License
See LICENSE file.
