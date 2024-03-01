import warnings
warnings.simplefilter(action='ignore')

from rdkit import Chem
from map4 import MAP4Calculator
import pandas as pd


NUM_DIMENSIONS = 2048
RADIUS = 1

map4 = MAP4Calculator(dimensions=NUM_DIMENSIONS, radius=RADIUS, is_counted=False, is_folded=True)

df = pd.read_json('../../../output/old_features/fusion.json', orient='index')
df.index.names = ['CID']

cids = df.index.values

fingerprints = {}

for cid in cids:
    row = df.loc[cid]

    smiles = row['IsomericSMILES']

    mol = Chem.MolFromSmiles(smiles)

    if mol.GetNumAtoms() == 1:
        fp_vector = [0] * NUM_DIMENSIONS
    else:
        fp_vector = map4.calculate(mol)
    
    fingerprints[cid] = [int(x) for x in fp_vector]

new_df = pd.DataFrame.from_dict(fingerprints, orient='index')
new_df.index.names = ['CID']
new_df = new_df.add_prefix('MAP4_').sort_index()

new_df.to_csv('../../../output/old_features/map4_fingerprints.csv')
