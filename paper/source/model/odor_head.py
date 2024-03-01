import tensorflow as tf
from tensorflow.keras import regularizers


class OdorHead(tf.keras.layers.Layer):
    def __init__(self, params, num_units):
        super(OdorHead, self).__init__()

        l2_factor = params['L2_factor']
        self.params = params
        self.N = params['N_H']
        self.ffn = []
        self.dropout = []
        self.norm = tf.keras.layers.LayerNormalization()

        self.stddev = 0.0
        
        for _ in range(self.N):
            self.ffn.append(tf.keras.layers.Dense(params['D_H'], activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=regularizers.l2(l2_factor)))
            self.dropout.append(tf.keras.layers.Dropout(params['P_Dropout_head']))
        
        self.final_layer = tf.keras.layers.Dense(num_units, activation=None, kernel_regularizer=regularizers.l2(l2_factor))


    def call(self, inp, tar, training, db_mask, skip_mask=False, skip_last_layer=False):
        if not skip_mask:
            binary_mask = tf.cast(tf.math.greater(db_mask, 0), dtype=tf.int32)

            masked_input = tf.boolean_mask(inp, binary_mask)
            masked_target = tf.boolean_mask(tar, binary_mask)
        else:
            masked_input = inp
            masked_target = tar

        if tf.shape(masked_input)[0] == 0:
            return masked_target, masked_target

        x = masked_input
        
        x = self.norm(x, training=training)

        for i in range(self.N):
            x = self.ffn[i](x)
            
            x = self.dropout[i](x, training=training)

        intermediate_features = x

        if skip_last_layer:
            output = intermediate_features
        else:
            output = self.final_layer(intermediate_features)

        return output, masked_target
