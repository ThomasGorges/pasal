import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

params = AllChem.ETKDGv3()
params.useSmallRingTorsions = True

mols = []

df = pd.read_json('../../../output/old_features/fusion.json', orient='index')

for cid in df.index.values:
    smiles = df.loc[cid, "IsomericSMILES"]
    
    mol = Chem.MolFromSmiles(smiles)
    
    if not mol:
        print('???')
        continue

    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
    mol.SetProp("_Name", smiles)

    mols.append(mol)

with Chem.SDWriter('../../../output/old_features/fusion.sdf') as w:
    for m in mols:
        w.write(m)

