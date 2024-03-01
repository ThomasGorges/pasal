import tensorflow as tf
from model.odor_head import OdorHead
from model.shared_model import SharedModel
from utility.score import denormalize_scores, calculate_scores_from_submission_in_memory
from utility.log import export_prediction_DREAM
import constants
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from model.loss_weighting import LossBalancing


def _unscale_dream(predictions):
    unscaled_predictions = {}

    for cid, pred in predictions.items():
        unscaled_values = constants.DREAM_SCALER.inverse_transform(np.array(pred).reshape(1, -1))
        unscaled_values = unscaled_values.tolist()[0]

        unscaled_predictions[cid] = unscaled_values

    return unscaled_predictions

def _average_predictions(last_predictions):
    avg_predictions = {}
    temp = {}

    # Reassemble structure
    for predictions in last_predictions:
        for cid in predictions:
            if cid not in temp:
                temp[cid] = []
            
            temp[cid].append(predictions[cid])
    
    # Average predictions
    for cid in temp:
         avg_predictions[cid.item()] = np.average(temp[cid], axis=0).tolist()
    
    #avg_predictions = _unscale_dream(avg_predictions)
    
    return avg_predictions

def _read_ground_truth(targets):
    return [
        targets[:, constants.DB_START_INDICES['dream_mean']:constants.DB_END_INDICES['dream_mean']],
        targets[:, constants.DB_START_INDICES['dream_std']:constants.DB_END_INDICES['dream_std']],
        targets[:, constants.DB_START_INDICES['arctander']:constants.DB_END_INDICES['arctander']],
        targets[:, constants.DB_START_INDICES['ifra_2019']:constants.DB_END_INDICES['ifra_2019']],
        targets[:, constants.DB_START_INDICES['leffingwell']:constants.DB_END_INDICES['leffingwell']],
        targets[:, constants.DB_START_INDICES['sigma_2014']:constants.DB_END_INDICES['sigma_2014']]
    ]

def _read_masks(db_identifiers):
    return [
        db_identifiers[:, 0],
        db_identifiers[:, 1],
        db_identifiers[:, 2],
        db_identifiers[:, 3],
        db_identifiers[:, 4],
        db_identifiers[:, 5],
    ]

class UniversalScent(tf.keras.Model):
    def __init__(self, summary_path, params, steps_per_epoch, loss_history_path, blacklisted_dbs=None):
        super(UniversalScent, self).__init__()

        self.params = params

        self.summary_path = summary_path
        self.epoch_count = params['Epochs']

        lr_schedule_head = tf.keras.optimizers.schedules.ExponentialDecay(params['LearningRate'], decay_steps=steps_per_epoch, decay_rate=0.99, staircase=True)
        lr_schedule_shared = tf.keras.optimizers.schedules.ExponentialDecay(params['LearningRate'], decay_steps=steps_per_epoch, decay_rate=0.99, staircase=True)

        self.optimizer_head = tfa.optimizers.AdamW(weight_decay=params['WeightDecay'], learning_rate=lr_schedule_head)
        self.optimizer_shared = tfa.optimizers.AdamW(weight_decay=params['WeightDecay'], learning_rate=lr_schedule_shared)

        blacklist_mask = np.zeros(constants.NUM_DATASETS, dtype=np.int32)
        if blacklisted_dbs:
            blacklisted_dbs = blacklisted_dbs.split(',')
            for db_idx in range(constants.NUM_DATASETS):
                if constants.DATASETS[db_idx] in blacklisted_dbs:
                    blacklist_mask[db_idx] = 1
        self.blacklist_mask = tf.constant(blacklist_mask)

        self.loss_weighting = LossBalancing(params, blacklist_mask)

        # Shared model
        self.shared_model = SharedModel(params)
        
        # Odor heads
        self.dream_mean_head = OdorHead(params, constants.NUM_ODORS['dream_mean'])
        self.dream_std_head = OdorHead(params, constants.NUM_ODORS['dream_std'])
        self.arctander_head = OdorHead(params, constants.NUM_ODORS['arctander'])
        self.ifra_head = OdorHead(params, constants.NUM_ODORS['ifra_2019'])
        self.leffingwell_head = OdorHead(params, constants.NUM_ODORS['leffingwell'])
        self.sigma_2014_head = OdorHead(params, constants.NUM_ODORS['sigma_2014'])

        self.__create_losses_and_accuracy()

        self.last_dream_val_z_score = 0.0


    def __create_losses_and_accuracy(self):
        self.loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.train_accuracies = {}
        self.train_losses = {}

        self.val_accuracies = {}
        self.val_losses = {}

        for db_name in constants.DATASETS:
            if db_name == 'dream_mean' or db_name == 'dream_std':
                self.train_accuracies[db_name] = tf.keras.metrics.MeanAbsoluteError(name=f'{db_name}_train_accuracy')
                self.val_accuracies[db_name] = tf.keras.metrics.MeanAbsoluteError(name=f'{db_name}_val_accuracy')
            else:
                self.train_accuracies[db_name] = tf.keras.metrics.BinaryAccuracy(name=f'{db_name}_train_accuracy')
                self.val_accuracies[db_name] = tf.keras.metrics.BinaryAccuracy(name=f'{db_name}_val_accuracy')

            self.train_losses[db_name] = tf.keras.metrics.Mean(name=f'{db_name}_train_loss')
            self.val_losses[db_name] = tf.keras.metrics.Mean(name=f'{db_name}_val_loss')


        self.loss_history = {
            'train': {},
            'val': {},
        }

        self.accuracy_history = {
            'train': {},
            'val': {
                'DREAM': []
            },
        }

        for db in constants.DATASETS:
            self.loss_history['train'][db] = []

            self.accuracy_history['train'][db] = []
    
    def call(self, features, training):
        return self.shared_model(features, training=training)

    step_signature=(
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        tf.TensorSpec(shape=(None, constants.NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None, constants.NUM_DATASETS), dtype=tf.int32),
        tf.TensorSpec(shape=(None, constants.TOTAL_CLASS_COUNT), dtype=tf.float32),
        tf.TensorSpec(shape=(constants.NUM_DATASETS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
    )
    @tf.function(input_signature=step_signature)
    def train_step(self, cids, features, combined_masks, merged_targets, weighting_factors, sparsity_ratio_temp, skip_head_optimization):
        batch_size = tf.shape(cids)[0]
        masks = _read_masks(combined_masks)
        ground_truth = _read_ground_truth(merged_targets)

        # Save gradients within this scope
        with tf.GradientTape(persistent=False) as tape:
            # Get shared model predictions
            general_prediction = self(features, True)

            # Get predictions for each head
            dream_mean_pred, dream_mean_tar = self.dream_mean_head(general_prediction, ground_truth[0], True, masks[0])
            dream_std_pred, dream_std_tar = self.dream_std_head(general_prediction, ground_truth[1], True, masks[1])
            arctander_pred, arctander_tar = self.arctander_head(general_prediction, ground_truth[2], True, masks[2])
            ifra_pred, ifra_tar = self.ifra_head(general_prediction, ground_truth[3], True, masks[3])
            leffingwell_pred, leffingwell_tar = self.leffingwell_head(general_prediction, ground_truth[4], True, masks[4])
            sigma_2014_pred, sigma_2014_tar = self.sigma_2014_head(general_prediction, ground_truth[5], True, masks[5])

            # This can be optimized, but prevents the reproducibility to previous results
            dream_mean_loss = self.loss_object(dream_mean_tar, dream_mean_pred)
            dream_std_loss = self.loss_object(dream_std_tar, dream_std_pred)
            arctander_loss = self.loss_object(arctander_tar, arctander_pred)
            ifra_loss = self.loss_object(ifra_tar, ifra_pred)
            leffingwell_loss = self.loss_object(leffingwell_tar, leffingwell_pred)
            sigma_2014_loss = self.loss_object(sigma_2014_tar, sigma_2014_pred)

            dream_mean_loss_v2 = self.loss_object(dream_mean_tar, dream_mean_pred)
            dream_std_loss_v2 = self.loss_object(dream_std_tar, dream_std_pred)
            arctander_loss_v2 = self.loss_object(arctander_tar, arctander_pred)
            ifra_loss_v2 = self.loss_object(ifra_tar, ifra_pred)
            leffingwell_loss_v2 = self.loss_object(leffingwell_tar, leffingwell_pred)
            sigma_2014_loss_v2 = self.loss_object(sigma_2014_tar, sigma_2014_pred)

            head_losses = [
                tf.reduce_mean(dream_mean_loss),
                tf.reduce_mean(dream_std_loss),
                tf.reduce_mean(arctander_loss),
                tf.reduce_mean(ifra_loss),
                tf.reduce_mean(leffingwell_loss),
                tf.reduce_mean(sigma_2014_loss)
            ]

            num_samples = [
                tf.shape(dream_mean_tar)[0],
                tf.shape(dream_std_tar)[0],
                tf.shape(arctander_tar)[0],
                tf.shape(ifra_tar)[0],
                tf.shape(leffingwell_tar)[0],
                tf.shape(sigma_2014_tar)[0]
            ]
            

            # Use filtered sample losses with pseudo labels for the shared model
            concat_loss = tf.concat(
                [
                    dream_mean_loss_v2 * weighting_factors[0],
                    dream_std_loss_v2 * weighting_factors[1],
                    arctander_loss_v2 * weighting_factors[2],
                    ifra_loss_v2 * weighting_factors[3],
                    leffingwell_loss_v2 * weighting_factors[4],
                    sigma_2014_loss_v2 * weighting_factors[5]
                ],
                axis=0
            )
            
            num_real_samples = tf.reduce_sum(num_samples)
            max_sample_count = tf.multiply(batch_size, constants.NUM_DATASETS - tf.reduce_sum(self.blacklist_mask))
            sparsity_ratio = tf.cast(num_real_samples, dtype=tf.float32) / tf.cast(max_sample_count, dtype=tf.float32)
            scale_ratio = tf.multiply(sparsity_ratio, tf.cast(batch_size, dtype=tf.float32))

            avg_loss_per_sample = tf.reduce_mean(concat_loss)

            batch_loss = tf.multiply(avg_loss_per_sample, scale_ratio)


            # Replace NaN with 0
            # Occurs when no samples are present in a head
            head_losses = tf.where(tf.math.is_nan(head_losses), 0.0, head_losses)

            weighted_losses = tf.multiply(head_losses, num_samples)
            balanced_head_losses = tf.multiply(weighted_losses, weighting_factors)

            gradients = tape.gradient(
                [
                    batch_loss,
                    balanced_head_losses[0],
                    balanced_head_losses[1],
                    balanced_head_losses[2],
                    balanced_head_losses[3],
                    balanced_head_losses[4],
                    balanced_head_losses[5],
                ], [
                    self.shared_model.trainable_variables,
                    self.dream_mean_head.trainable_variables,
                    self.dream_std_head.trainable_variables,
                    self.arctander_head.trainable_variables,
                    self.ifra_head.trainable_variables,
                    self.leffingwell_head.trainable_variables,
                    self.sigma_2014_head.trainable_variables
                ],
                unconnected_gradients=tf.UnconnectedGradients.ZERO
            )

        if skip_head_optimization:
            self.optimizer_shared.apply_gradients(zip([
                *gradients[0]
            ], [
                *self.shared_model.trainable_variables
            ]))
        else:
            self.optimizer_head.apply_gradients(zip([
                *gradients[1],
                *gradients[2],
                *gradients[3],
                *gradients[4],
                *gradients[5],
                *gradients[6]
            ], [
                *self.dream_mean_head.trainable_variables,
                *self.dream_std_head.trainable_variables,
                *self.arctander_head.trainable_variables,
                *self.ifra_head.trainable_variables,
                *self.leffingwell_head.trainable_variables,
                *self.sigma_2014_head.trainable_variables
            ]))

        if not skip_head_optimization:
            self.track_train_performance(weighting_factors,
                dream_mean_tar, dream_mean_pred,
                dream_std_tar, dream_std_pred,
                arctander_tar, arctander_pred,
                ifra_tar, ifra_pred,
                leffingwell_tar, leffingwell_pred,
                sigma_2014_tar, sigma_2014_pred)


    def track_train_performance(self, weighting_factors,
            dream_mean_tar, dream_mean_pred,
            dream_std_tar, dream_std_pred,
            arctander_tar, arctander_pred,
            ifra_tar, ifra_pred,
            leffingwell_tar, leffingwell_pred,
            sigma_2014_tar, sigma_2014_pred
        ):
        # Cast predictions to 0 or 1
        ifra_pred_binary = tf.cast(tf.sigmoid(ifra_pred) > 0.5, dtype=tf.int32)
        arctander_pred_binary = tf.cast(tf.sigmoid(arctander_pred) > 0.5, dtype=tf.int32)
        leffingwell_pred_binary = tf.cast(tf.sigmoid(leffingwell_pred) > 0.5, dtype=tf.int32)
        sigma_2014_pred_binary = tf.cast(tf.sigmoid(sigma_2014_pred) > 0.5, dtype=tf.int32)

        if tf.shape(dream_mean_tar)[0] > 0:
            self.train_losses['dream_mean'](weighting_factors[0] * self.loss_object(dream_mean_tar, dream_mean_pred))
            self.train_accuracies['dream_mean'](dream_mean_tar, dream_mean_pred)
        
        if tf.shape(dream_std_tar)[0] > 0:
            self.train_losses['dream_std'](weighting_factors[1] * self.loss_object(dream_std_tar, dream_std_pred))
            self.train_accuracies['dream_std'](dream_std_tar, dream_std_pred)

        if tf.shape(arctander_tar)[0] > 0:
            self.train_losses['arctander'](weighting_factors[2] * self.loss_object(arctander_tar, arctander_pred))
            self.train_accuracies['arctander'](arctander_tar, arctander_pred_binary)

        if tf.shape(ifra_tar)[0] > 0:
            self.train_losses['ifra_2019'](weighting_factors[3] * self.loss_object(ifra_tar, ifra_pred))
            self.train_accuracies['ifra_2019'](ifra_tar, ifra_pred_binary)

        if tf.shape(leffingwell_tar)[0] > 0:
            self.train_losses['leffingwell'](weighting_factors[4] * self.loss_object(leffingwell_tar, leffingwell_pred))
            self.train_accuracies['leffingwell'](leffingwell_tar, leffingwell_pred_binary)

        if tf.shape(sigma_2014_tar)[0] > 0:
            self.train_losses['sigma_2014'](weighting_factors[5] * self.loss_object(sigma_2014_tar, sigma_2014_pred))
            self.train_accuracies['sigma_2014'](sigma_2014_tar, sigma_2014_pred_binary)

    def train(self, datasets, silent=False, old_model=None):
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        test_dataset = datasets['test']

        self.loss_weighting.set_num_samples(train_dataset.num_mols_per_db)

        # Save predictions of latest epochs and average them to get a more stable result
        last_val_predictions = []
        last_test_predictions = []

        sparsity_ratio = tf.convert_to_tensor(train_dataset.sparsity_ratio, dtype=tf.float32)

        for epoch in range(self.epoch_count):
            # Reset all trackers
            for trackers in [self.train_accuracies, self.train_losses, self.val_accuracies, self.val_losses]:
                for v in trackers.values():
                    v.reset_states()
            
            loss_weights = self.loss_weighting.get_loss_weights()

            for batch in train_dataset.tf_dataset:
                self.train_step(*batch, loss_weights, sparsity_ratio, tf.constant(False, dtype=tf.bool))

                # Dont update shared model at last epoch
                if epoch != self.epoch_count - 1:
                    self.train_step(*batch, loss_weights, sparsity_ratio, tf.constant(True, dtype=tf.bool))
            
            self.loss_weighting.track_losses(self.train_losses)
            self.loss_weighting.update(train_dataset.phi)

            val_dream_predictions = self.predict_dream(val_dataset)
            if len(last_val_predictions) == 7:
                last_val_predictions.pop(0)
            last_val_predictions.append(val_dream_predictions)

            test_dream_predictions = self.predict_dream(test_dataset)
            if len(last_test_predictions) == 7:
                last_test_predictions.pop(0)
            last_test_predictions.append(test_dream_predictions)

            avg_val_predictions = _average_predictions(last_val_predictions)
            avg_test_predictions = _average_predictions(last_test_predictions)

            val_scores = calculate_scores_from_submission_in_memory(val_dataset.dream_cid_order, val_dataset.dream_labels, avg_val_predictions)
            test_scores = calculate_scores_from_submission_in_memory(constants.GS_CID_ORDER, constants.GS_GROUND_TRUTH, avg_test_predictions)

            if not silent:
                print(f'DREAM val: {val_scores["z_score"]} | {float(val_scores["mse_loss"])}')
                print(f'DREAM test: {test_scores["z_score"]} | {float(test_scores["mse_loss"])}')

                loss_msg = ''
                for k in self.train_losses.keys():
                    if loss_msg:
                        loss_msg += ' | '
                    loss_msg += f'{k} ({self.train_losses[k].result():.3f}|{self.train_accuracies[k].result():.3f})'
                print(f'T #{epoch + 1:3}: (Losses|Acc) | ' + loss_msg)

            # Save tracker history
            # astype(float) is needed to serialize those values to JSON. np.float32 won't work
            for db in constants.DATASETS:
                self.loss_history['train'][db].append(self.train_losses[db].result().numpy().astype(float))
                self.accuracy_history['train'][db].append(self.train_accuracies[db].result().numpy().astype(float))
            
            self.accuracy_history['val']['DREAM'].append(val_scores['z_score'])

        if not silent:
            features_df = {
                "leffingwell": self.predict_features(
                    datasets.values(), specific_head="leffingwell"
                ),
                "ifra_2019": self.predict_features(
                    datasets.values(), specific_head="ifra_2019"
                ),
                "shared_model": self.predict_features(datasets.values()),
            }

            for k, v in features_df.items():
                v.to_csv(self.summary_path + f"/features_{k}_perspective.csv")

            export_prediction_DREAM(self.summary_path + "/prediction_LBs2.txt", avg_val_predictions)
            export_prediction_DREAM(self.summary_path + "/prediction_GSs2.txt", avg_test_predictions)
        
        val_scores['val_predictions'] = avg_val_predictions#_unscale_dream(avg_val_predictions)
        val_scores['test_predictions'] = avg_test_predictions#_unscale_dream(avg_test_predictions)

        return val_scores
    

    def predict_features(self, datasets, specific_head=None):
        predictions = []
        num_features = -1

        processed_cids = []

        for dataset in datasets:
            for cids, features, _, _ in dataset.tf_dataset:
                features = self(features, False)

                if specific_head == "ifra_2019":
                    features, _ = self.ifra_head(
                        features, None, False, None, True, skip_last_layer=True
                    )
                elif specific_head == "leffingwell":
                    features, _  = self.leffingwell_head(
                        features, None, False, None, True, skip_last_layer=True
                    )

                num_samples = tf.shape(features)[0]

                if num_features == -1:
                    num_features = tf.shape(features)[1].numpy().item()

                for idx in range(num_samples):
                    cid = cids[idx].numpy()[0]

                    if cid in processed_cids:
                        continue

                    processed_cids.append(cid)

                    mol_features = features[idx].numpy().tolist()

                    # prediction = denormalize_scores(prediction)
                    mol_features = [cid] + mol_features

                    predictions.append(tuple(mol_features))

        header = [f"FEATURE_{i}" for i in range(num_features)]
        df = pd.DataFrame(predictions, columns=["CID"] + header)
        df = df.set_index("CID").sort_index()

        return df


    def get_dream_val_loss(self):
        return self.val_losses["DREAM"].result().numpy()
    

    def get_dream_val_z_score(self):
        return self.last_dream_val_z_score


    def predict_general_prediction(self, features):
        return self(features, False)

    def predict(self, input):
        general_predictions = self.predict_general_prediction(input)

        predictions, _ = self.dream_head(general_predictions, None, False, None, skip_mask=True)

        return predictions.numpy()
    
    def predict_dream(self, test_dataset):
        predictions = {}

        for cids, features, masks, _ in test_dataset.tf_dataset:
            masks = _read_masks(masks)
            
            is_in_dream_mean = masks[0] == 1
            is_in_dream_std = masks[1] == 1

            features = self.predict_general_prediction(features)
            num_samples = tf.shape(features)[0]

            dream_mean_pred, _  = self.dream_mean_head(features, None, False, None, True)
            dream_std_pred, _ = self.dream_std_head(features, None, False, None, True)
            
            for idx in range(num_samples):
                if not is_in_dream_mean[idx] or not is_in_dream_std[idx]:
                    continue

                cid = cids[idx].numpy()[0]

                mol_mean_pred = dream_mean_pred[idx].numpy().tolist()
                mol_std_pred = dream_std_pred[idx].numpy().tolist()

                mol_pred = []

                for i in range(42):
                    if i % 2 == 0:
                        mol_pred.append(mol_mean_pred[i // 2])
                    else:
                        mol_pred.append(mol_std_pred[(i - 1) // 2])

                predictions[cid] = mol_pred
        
        return predictions

    def predict_dream_valset_and_export(self, dataset):
        predictions = self.predict_dream(dataset)

        export_prediction_DREAM(self.summary_path + f'/prediction_LBs2.txt', predictions)


    def predict_dream_testset_and_export(self, dataset):
        predictions = self.predict_dream(dataset)

        export_prediction_DREAM(self.summary_path + f'/prediction_GSs2.txt', predictions)

