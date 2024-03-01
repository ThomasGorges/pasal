import pandas as pd
import json
import constants


def load_classes(filename):
    classes = {}

    df = pd.read_json(filename, orient='index')

    for db_name in df.index.values:
        if db_name in ['aromadb', 'flavordb', 'flavornet']:
            continue

        db_classes = df.loc[db_name].dropna().values
        
        classes[db_name] = db_classes
    
    return classes

def calculate_num_classes(odors):
    num_classes = {
        'dream_mean': 21,
        'dream_std': 21
    }

    for k, v in odors.items():
        num_classes[k] = len(v)
    
    return num_classes


def load_cooccurrence_data(filename):
    with open(filename) as f:
        data = json.load(f)
    
    return data

def calculate_db_indices(datasets, pyrfume_odors):
    start_indices, end_indices = {}, {}

    start_idx = 0
    end_idx = start_idx

    for db_name in datasets:
        if db_name == 'dream_mean' or db_name == 'dream_std':
            num_classes = 21
        else:
            num_classes = len(pyrfume_odors[db_name])
        
        end_idx += num_classes
        
        start_indices[db_name] = start_idx
        end_indices[db_name] = end_idx

        start_idx = end_idx
    
    return start_indices, end_indices