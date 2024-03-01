import pandas as pd
import numpy as np
import random
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem


df = pd.read_json('../../../output/old_features/fusion.json', orient='index')
df.index.names = ['CID']

np.random.seed(0)
random.seed(0)

# Calculate physicochemical properties
calc = Calculator(descriptors, ignore_3D=False)

print(f'Number of descriptors: {len(calc.descriptors)}')

params = AllChem.ETKDGv3()
params.useSmallRingTorsions = True

molecular_features = {}

for cid in df.index.values:
    smiles = df.loc[cid]['IsomericSMILES']

    mol = Chem.MolFromSmiles(smiles)
    
    if not mol:
        print('???')
        continue

    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)

    molecular_features[cid] = calc(mol)

new_df = pd.DataFrame.from_dict(molecular_features, orient='index', columns=calc.descriptors)
new_df.index.names = ['CID']
new_df.to_csv('../../../output/old_features/mordred.csv')
