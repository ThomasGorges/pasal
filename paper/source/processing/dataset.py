import itertools
import numpy as np
import pandas as pd
import constants
import tensorflow as tf


class Dataset:
    def __init__(self, db_type='train', fold_id=-1, appendix_path=None, exclude_list=None):
        self.appendix_path = appendix_path
        self.db_blacklist = []

        self.blacklist_mask = np.zeros(constants.NUM_DATASETS, dtype=np.bool)
        if exclude_list:
            self.db_blacklist = exclude_list.split(',')

            for db_name in self.db_blacklist:
                if db_name in self.db_blacklist:
                    self.blacklist_mask[constants.DATASETS.index(db_name)] = True

        self.tf_dataset = None
        self.pseudo_labels = {}
        self.db_type = db_type
        if self.db_type in ['train', 'val']:
            self.filename = f'../output/preprocessing/train_val_set'
        else:
            self.filename = f'../output/preprocessing/{db_type}_set'
        self.steps_per_epoch = None
        self.batch_size = None
        self.fold_id = int(fold_id)


    def load(self, batch_size, shuffle=False):
        self.batch_size = batch_size

        dataset = pd.read_json(f'{self.filename}.json', orient='index')
        dataset.index.name = 'CID'

        if self.appendix_path:
            appendix_df = pd.read_json(self.appendix_path, orient='index')
            appendix_df.index.name = 'CID'

            dataset = pd.concat([dataset, appendix_df])

        if self.fold_id != -1:
            fold_cids_df = pd.read_json('../output/preprocessing/train_folds.json', orient='index')
            fold_cids_df.index.names = ['fold']
            self.fold_cids = fold_cids_df.loc[self.fold_id]
        else:
            self.fold_cids = None

        self.features_df = pd.read_csv('../output/preprocessing/reduced_features.csv').set_index('CID')

        labels = dataset.drop(columns=['IsomericSMILES'])
        self.dataset = labels

        cids, features, db_identifiers, odor_data = self._process_labels(labels, self.features_df)

        return self._to_tf_dataset(cids, features, db_identifiers, odor_data, shuffle)


    def _to_tf_dataset(self, cids, features, db_identifiers, labels, shuffle):
        padded_shapes = ([-1], [-1], [-1], [-1])

        self.dream_labels = {}
        self.dream_cid_order = []

        for i in range(len(cids)):
            if db_identifiers[i][constants.DATASETS.index('dream_mean')] == 0:
                continue

            cid = cids[i][0]

            # Skip oversampled molecules
            if cid in self.dream_labels:
                continue
            self.dream_cid_order.append(str(cid))
            
            # Source order: dream mean (N = 21), dream std (N = 21)
            # Taget order: dream mean[0], dream std[0], dream mean[1], ...
            mol_dream_labels = labels[i][:42]
            self.dream_labels[str(cid)] = {}

            for j in range(21):
                dream_class = constants.DREAM_ODORS[j]
                self.dream_labels[str(cid)][dream_class] = (mol_dream_labels[j], mol_dream_labels[j + 21])

        dataset = tf.data.Dataset.from_tensor_slices((cids, features, db_identifiers, labels))

        if shuffle:
            dataset = dataset.shuffle(len(cids), reshuffle_each_iteration=True)

        if self.batch_size == -1:
            self.batch_size = len(cids)

        dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)

        self.steps_per_epoch = np.ceil(len(cids) / self.batch_size).astype(int)

        self.tf_dataset = dataset.cache()


    def _process_labels(self, labels, features_df):
        cids = []
        db_identifiers = []
        odor_data = []
        pca_features = []

        unique_cids = labels.index.values
        
        for cid in unique_cids:
            if self.fold_cids is not None:
                if cid not in self.fold_cids[self.db_type + '_cids']:
                    continue

            mol = labels.loc[cid]
            raw_odors = mol.values

            self.pseudo_labels[cid] = []

            for blacklisted_db in self.db_blacklist:
                mol = mol.drop(blacklisted_db)

            # Molecules only exist in blacklisted datasets -> skip
            if mol.dropna().shape[0] == 0:
                continue

            pca_features.append(features_df.loc[cid])

            mol_identifiers = np.zeros(constants.NUM_DATASETS)

            processed_odors = []

            has_dream = False
            for db_idx in range(constants.NUM_DATASETS):
                db_name = constants.DATASETS[db_idx]
                
                if db_name in self.db_blacklist:
                    db_blacklisted = True
                else:
                    db_blacklisted = False

                if not db_name in labels.columns or db_blacklisted:
                    label_exists = False
                else:
                    adjusted_idx = labels.columns.values.tolist().index(db_name)

                    # Check if we have values for the specific molecule and dataset
                    label_exists = False
                    if raw_odors[adjusted_idx] is not None:
                        label_exists = True

                        # We have one value, it might be NaN
                        if isinstance(raw_odors[adjusted_idx], float):
                            if np.isnan(raw_odors[adjusted_idx]):
                                label_exists = False

                if label_exists:
                    mol_identifiers[db_idx] = 1

                if not mol_identifiers[db_idx]:
                    num_odors = constants.NUM_ODORS[db_name]

                    processed_odors.append(np.zeros(num_odors, dtype=np.float32))
                else:
                    data_source = raw_odors[adjusted_idx]

                    if db_name in ['dream_mean', 'dream_std']:
                        ds_odors = np.array(data_source, dtype=np.float32)
                        has_dream = True
                    else:
                        ds_odors = self._labels_to_hot_encoding(data_source, db_name)

                    processed_odors.append(ds_odors)

            # Oversample DREAM train molecules
            if has_dream and self.db_type == 'train':
                cids.append(cid)
                db_identifiers.append(mol_identifiers)
                odor_data.append(list(itertools.chain(*processed_odors)))
                pca_features.append(features_df.loc[cid])

            cids.append(cid)
            db_identifiers.append(mol_identifiers)
            odor_data.append(list(itertools.chain(*processed_odors)))
        
        pca_features = np.array(pca_features)
        odor_data = np.array(odor_data)

        num_total_mols = len(cids)
        self.num_mols_per_db = np.count_nonzero(np.array(db_identifiers), axis=0)
        self.phi = self.num_mols_per_db / num_total_mols

        self.phi = self.phi[~self.blacklist_mask]

        cids = np.array(cids).reshape((-1, 1))
        db_identifiers = np.array(db_identifiers)

        cids = cids.astype(np.int32)
        pca_features = pca_features.astype(np.float32)
        db_identifiers = db_identifiers.astype(np.int32)
        odor_data = odor_data.astype(np.float32)

        self.sparsity_ratio = np.count_nonzero(db_identifiers) / (len(cids) * (constants.NUM_DATASETS - len(self.db_blacklist)))

        return cids, pca_features, db_identifiers, odor_data

    def _labels_to_hot_encoding(self, labels, db_name):
        classes = constants.PYRFUME_ODORS[db_name]

        v = [0.0] * len(classes)

        for label in labels:
            for i in range(len(classes)):
                if label == classes[i]:
                    v[i] = 1.0

        if self.db_type == 'test':
            return np.array(v, dtype=np.float32)

        # Use knowledge from cooccurence matrix and punish impossible combinations
        # We build a list and remember all the impossible combinations for that specific molecule
        merged_known_combinations = []
        for label in labels:
            known_combinations = constants.COOCCURRENCE_DATA[db_name][label]

            # We need to add our current label in the merged list
            for known_odor in known_combinations:
                if known_odor not in merged_known_combinations:
                    merged_known_combinations.append(known_odor)


        for odor_idx in range(len(classes)):
            if classes[odor_idx] in labels:
                continue

            if classes[odor_idx] not in merged_known_combinations:
                v[odor_idx] = -0.15

        return np.array(v, dtype=np.float32)