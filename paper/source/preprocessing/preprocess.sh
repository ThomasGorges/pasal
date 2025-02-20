echo fusion...
python3 fusion.py

echo splitting data...
python3 split_dataset.py

echo extracting classes...
python3 extract_classes.py

echo co-occurrence calculation...
python3 calculate_known_combinations.py

echo scaling DREAM LBs2...
python3 scale_dream_lbs2.py

echo calculating features...
python3 feature_calculator.py

echo selecting features...
python3 feature_selection.py

echo feature reduction...
python3 feature_reduction.py

echo calculating embeddings...
python3 extract_molformer_features.py

echo analyzing train split...
python3 analyze_trainset.py
