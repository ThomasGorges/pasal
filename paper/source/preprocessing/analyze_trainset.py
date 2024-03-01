import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


df = pd.read_json('../output/preprocessing/train_set.json', orient='index')
df = df.drop(columns=['IsomericSMILES'])

num_datasets = len(df.columns)

print(f'N total molecules: {df.shape[0]}')

partial_df = pd.DataFrame()
db_names = ['dream_mean', 'sigma_2014', 'ifra_2019', 'arctander', 'leffingwell']

for db_name in db_names:
    partial_df[db_name] = df[db_name]

    db_df = df[db_name].dropna(axis=0, how='all')

    N_db = db_df.shape[0]
    N_classes = 0

    if db_name == 'dream_mean':
        N_classes = 21
    else:
        mlb = MultiLabelBinarizer()
        mlb.fit(db_df)
        N_classes = len(mlb.classes_)

    temp_df = partial_df.dropna(axis=0, how='all')

    if len(partial_df.columns) > 1:
        print('+ ', end='')
    
    sparsity = temp_df.isna().sum().sum() / (temp_df.shape[0] * temp_df.shape[1])

    
    print(f'{db_name}: N={N_db}, N classes={N_classes}, (Total: N={temp_df.shape[0]}, sparisty={sparsity * 100.0:.2f})')
