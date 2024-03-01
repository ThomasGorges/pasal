from rdkit import Chem
from requests import Session
import os
import time
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# SMILES returned from PubChem seem to not use the canonical SMILES from RDKit
# This function generates a canonical SMILES with RDKit
def get_canonical_smiles(smiles, explicit_bonds=False, explicit_h_atoms=False):
    mol = Chem.MolFromSmiles(smiles)

    rdMolStandardize.Cleanup(mol)

    canonical_smiles = Chem.MolToSmiles(mol,
                                        canonical=True,
                                        isomericSmiles=True,
                                        allBondsExplicit=explicit_bonds,
                                        allHsExplicit=explicit_h_atoms)

    return canonical_smiles


def merge_dicts(dicts):
    merged_dict = {}

    for d in dicts:
        for k, v in d.items():
            merged_dict[k] = v

    return merged_dict


class PubChem:
    def __init__(self):
        self._last_request_timestamp = 0
        self._session = Session()
        self._session.headers.update({'Connection': 'Keep-Alive'})
        self._pubchem_base_url = 'https://pubchem.ncbi.nlm.nih.gov/'

        self._cache = self._load_cache('cache.txt')

    def _load_cache(self, filename):
        cached_values = {}

        if not os.path.exists(filename):
            return cached_values

        with open(filename) as input:
            lines = input.readlines()

        for line in lines:
            if line.startswith('#'):
                continue

            seperator_idx = line.index(': ')
            v = line[seperator_idx + len(': '):].strip()
            k = line[:seperator_idx]

            cached_values[k] = v

        return cached_values

    def _update_cache(self, filename, k, v):
        self._cache[k] = v
        
        with open(filename, 'a') as output:
            output.write(f'{k}: {v}\n')

    def fetch_smiles_from_cid(self, cid):
        smiles = self._cached_fetch(cid, 'property/IsomericSMILES', 'cid')

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            rdMolStandardize.Cleanup(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        
        return smiles

    def fetch_smiles(self, name_or_cid):
        smiles = self._cached_fetch(name_or_cid, 'property/IsomericSMILES')

        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            rdMolStandardize.Cleanup(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        
        return smiles

    def fetch_cid_from_name(self, name):
        return self._cached_fetch(name, 'cids')

    def _cached_fetch(self, name, query, name_type='name'):
        k = str(name_type) + '/' + str(name) + '/' + str(query)
        if not self._cache.get(k, ''):
            v = self._fetch(name, query, name_type)

            if v:
                self._update_cache('cache.txt', k, v)
        else:
            v = self._cache[k]

        return v

    def _fetch(self, name, query, name_type='name'):
        # Fetch SMILES from PubChem by CID
        request_uri = self._pubchem_base_url + f'/rest/pug/compound/{name_type}/{name}/{query}/TXT'

        response = ''

        while not response:
            try:
                if self._last_request_timestamp:
                    while time.time() - self._last_request_timestamp < 0.2:
                        time.sleep(0.001)

                response = self._session.get(request_uri).text
                self._last_request_timestamp = time.time()
            except:
                response = ''

        smiles = response.split('\n')[0]

        if 'Status:' in smiles:
            return ''

        return smiles
