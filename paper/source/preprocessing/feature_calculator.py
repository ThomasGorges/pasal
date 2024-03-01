import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from mordred import Calculator, descriptors
from padelpy import padeldescriptor
from map4 import MAP4Calculator
import tempfile
import os
from joblib import dump


def preprocess_conformers(conformers, cid_list=None):
    mols = []

    for cid, mol in conformers.items():
        mol.SetProp("_Name", str(cid))

        if cid_list is not None and int(cid) not in cid_list:
            continue

        mols.append(mol)

    return mols


def filter_cids(molecules, cid_list):
    filtered_mols = []

    for mol in molecules:
        cid = int(mol.GetProp('_Name'))

        if cid in cid_list:
            filtered_mols.append(mol)

    return filtered_mols


def mols_to_sdf(molecules, sdf_path=None):
    if not sdf_path:
        sdf_file = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False)
        sdf_file.close()
        sdf_path = sdf_file.name

    with Chem.SDWriter(sdf_path) as w:
        for m in molecules:
            w.write(Chem.RemoveHs(m))

    return sdf_path


def calculate_padel_features(sdf_path, calculate_3d=True):
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    temp_file.close()

    try:
        padeldescriptor(mol_dir=sdf_path,
            d_file=temp_file.name,
            detectaromaticity=True,
            d_2d=True,
            d_3d=calculate_3d,
            fingerprints=False,
            maxruntime=60000,
            removesalt=True,
            retain3d=True,
            retainorder=True,
            standardizenitro=False,
            standardizetautomers=False
        )
    except:
        pass

    padel_df = pd.read_csv(temp_file.name)
    padel_df = padel_df.rename(columns={'Name': 'CID'})
    padel_df = padel_df.set_index('CID')

    os.unlink(temp_file.name)

    return padel_df


def calculate_mordred_features(conformers):
    mordred_calc = Calculator(descriptors, ignore_3D=False)

    mordred_features = {}
    for cid, mol in conformers.items():
        mordred_features[cid] = mordred_calc(mol)

    mordred_df = pd.DataFrame.from_dict(
        mordred_features, orient="index", columns=mordred_calc.descriptors
    )
    mordred_df.index.names = ["CID"]

    return mordred_df


def calculate_foldedAP2(molecules, map4=False):
    if map4:
        foldedAP2 = MAP4Calculator(dimensions=1024, radius=2)
    else:
        foldedAP2 = MAP4Calculator(dimensions=2048, radius=1, is_counted=False, is_folded=True)

    fusion_df = pd.read_json('../../output/preprocessing/fusion.json', orient='index')
    fusion_df.index.names = ['CID']

    foldedAP2_features = {}
    for mol in molecules:
        cid = int(mol.GetProp('_Name'))

        old_smiles = fusion_df.loc[cid]['IsomericSMILES']

        new_mol = Chem.MolFromSmiles(old_smiles)

        if new_mol.GetNumAtoms() == 1:
            fp_vector = [0] * 2048
        else:
            fp_vector = foldedAP2.calculate(new_mol)

        foldedAP2_features[cid] = [int(x) for x in fp_vector]

    foldedAP2_df = pd.DataFrame.from_dict(foldedAP2_features, orient="index")
    foldedAP2_df.index.names = ["CID"]

    if map4:
        foldedAP2_df = foldedAP2_df.add_prefix("MAP4_")
    else:
        foldedAP2_df = foldedAP2_df.add_prefix("foldedAP2_")

    return foldedAP2_df


def calculate_all_features(conformers, cid_list=None, sdf_path=None, return_molecules=False, calculate_3d=True):
    molecules = preprocess_conformers(conformers, cid_list)

    sdf_path = mols_to_sdf(molecules, sdf_path)

    print('Calculating PaDEL features...')
    padel_df = calculate_padel_features(sdf_path, calculate_3d)

    print('Calculating mordred features...')
    mordred_df = calculate_mordred_features(conformers)

    # Write and load mordred features to convert column names to string
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=True)
    mordred_df.to_csv(temp_file.name)
    mordred_df = pd.read_csv(temp_file.name).set_index('CID')
    temp_file.close()

    print('Calculating foldedAP2 features...')
    foldedAP2_df = calculate_foldedAP2(molecules)

    if return_molecules:
        return (padel_df, mordred_df, foldedAP2_df, molecules)
    else:
        return (padel_df, mordred_df, foldedAP2_df)


def merge_features(padel_df, mordred_df, foldedAP2_df, mordred_cols_to_drop=None, add_prefix=False, fill_nan_with_zero=False):
    if mordred_cols_to_drop:
        mordred_df = mordred_df.drop(columns=mordred_cols_to_drop)

    if add_prefix:
        padel_df = padel_df.add_prefix('PaDEL_')
        mordred_df = mordred_df.add_prefix('mordred_')

    combined_df = padel_df.join(mordred_df, on='CID').join(foldedAP2_df, on='CID')

    return combined_df


if __name__ == '__main__':
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.randomSeed = 0
    
    conformers = {}
    
    df = pd.read_json('../../output/preprocessing/fusion.json', orient='index')
    
    for cid in tqdm(df.index.values, desc='Molecules'):
        smiles = df.loc[cid, "IsomericSMILES"]
    
        mol = Chem.MolFromSmiles(smiles)
    
        if not mol:
            print('???')
            continue
    
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
    
        conformers[cid] = mol
    
    print(len(conformers.keys()))
    dump(conformers, '../../output/preprocessing/conformers.joblib')

    padel_df, mordred_df, foldedAP2_df, molecules = calculate_all_features(conformers, return_molecules=True, calculate_3d=True)
    padel_df.to_csv('../../output/preprocessing/features_padel.csv')
    mordred_df.to_csv('../../output/preprocessing/features_mordred.csv')
    foldedAP2_df.to_csv('../../output/preprocessing/features_foldedAP2.csv')

    all_descriptors = list(set([*padel_df.columns] + [*mordred_df.columns]))

    mordred_features_to_drop = []

    for desc in all_descriptors:
        if desc in padel_df.columns and desc in mordred_df.columns:
            mordred_features_to_drop.append(desc)

    merged_df = merge_features(padel_df, mordred_df, foldedAP2_df, mordred_cols_to_drop=mordred_features_to_drop, add_prefix=False, fill_nan_with_zero=False)

    merged_df.to_csv('../../output/preprocessing/features_merged.csv')
