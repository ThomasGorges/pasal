import tensorflow as tf
import numpy as np
import constants


class LossBalancing:
    def __init__(self, params, blacklist_mask, use_constant=False):
        self.weight_factors = {}
        self.loss_history = {}
        
        self.unscaled_losses = {}

        self.alpha = params['Alpha']
        self.beta = params['Beta']

        if self.alpha == 0.0 and self.beta == 0.0:
            use_constant = True
        
        self.blacklist_mask = blacklist_mask
        self.r_history = {}
        self.p_history = {}
        self.lambda_history = {}
        self.lambda_history_scaled = {}
        self.l_hat_history = []

        for idx in range(constants.NUM_DATASETS):
            if self.blacklist_mask[idx] != 0:
                continue

            k = constants.DATASETS[idx]

            self.weight_factors[k] = 1.0
            
            self.r_history[k] = []
            self.p_history[k] = []
            self.lambda_history[k] = []
            self.lambda_history_scaled[k] = []

            self.unscaled_losses[k] = []

            self.loss_history[k] = []

        self.use_constant = use_constant
        if use_constant:
            self.weight_factors = {
                'dream_mean': 1.60,
                'dream_std': 1.13,
                'arctander': 0.78,
                'ifra_2019': 0.72,
                'leffingwell': 1.14,
                'sigma_2014': 0.63
            }


        self.epoch = 0

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def track_losses(self, losses):
        for k in losses.keys():
            db_idx = constants.DATASETS.index(k)

            if self.blacklist_mask[db_idx] != 0:
                continue

            self.loss_history[k].append(losses[k].result().numpy())

    def update(self, phi):
        # Track losses
        last_losses = []

        for k in self.loss_history.keys():
            last_losses.append(self.loss_history[k][-1])

        # Calculate average sample-loss based on head losses
        l_hat = np.average(last_losses, weights=phi)

        # Stabilize l_hat at first epoch
        if self.epoch == 0:
            l_hat = 1.0 / len(self.weight_factors.keys())
        
        self.l_hat_history.append(l_hat)

        # Local learning progress p
        p = {}
        for k in self.unscaled_losses.keys():
            # In first epoch weight_factors[k] == 1
            self.unscaled_losses[k].append(
                self.loss_history[k][-1] / self.weight_factors[k]
            )

            p[k] = self.unscaled_losses[k][-1] / self.unscaled_losses[k][0]

        # Relative inverse training rate
        r = {}
        for k in self.unscaled_losses.keys():
            r[k] = (p[k] * np.sum(phi)) / np.sum(phi * list(p.values()))

        for k in self.loss_history.keys():
            if not self.use_constant:
                self.weight_factors[k] = self.weight_factors[k] * self.beta + 0.5 * (1.0 - self.beta) * (1.0 + l_hat / (self.loss_history[k][-1] * np.power(r[k], self.alpha)))
            
            self.r_history[k].append(r[k])
            self.p_history[k].append(p[k])

            self.lambda_history[k].append(self.weight_factors[k])
            self.lambda_history_scaled[k].append(self.weight_factors[k] * self.num_samples[constants.DATASETS.index(k)])

        self.epoch += 1

    def get_loss_weights(self):
        # Convert dict to list in the correct order
        loss_weights = []

        for idx in range(constants.NUM_DATASETS):
            if self.blacklist_mask[idx] != 0:
                loss_weights.append(0.0)
            else:
                db_name = constants.DATASETS[idx]
                loss_weights.append(self.weight_factors[db_name])

        loss_weights = tf.convert_to_tensor(loss_weights, dtype=tf.float32)

        return loss_weights

    def get_history(self):
        return {
            "lambda_history": self.lambda_history,
            "lambda_history_scaled": self.lambda_history_scaled,
            "l_hat": self.l_hat_history,
            "r": self.r_history,
            "p": self.p_history
        }
