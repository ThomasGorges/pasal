import tensorflow as tf
from tensorflow.keras import regularizers


class SharedModel(tf.keras.layers.Layer):
    def __init__(self, params):
        super(SharedModel, self).__init__()

        l2_factor = params['L2_factor']
        self.N = params['N_Shared']

        self.dense = [
            tf.keras.layers.Dense(params['D_Shared_first'], activation='relu', kernel_regularizer=regularizers.l2(l2_factor)),
            tf.keras.layers.Dense(params['D_Shared_intermediate'], activation='relu', kernel_regularizer=regularizers.l2(l2_factor)),
        ]

        for _ in range(self.N):
            self.dense.append(tf.keras.layers.Dense(params['D_Shared_intermediate'], activation='relu', kernel_regularizer=regularizers.l2(l2_factor)))
        
        self.dense.append(tf.keras.layers.Dense(params['D_Shared_last'], activation='relu', kernel_regularizer=regularizers.l2(l2_factor)))

        self.dropout = [
            tf.keras.layers.Dropout(params['P_Dropout_shared'])
        ]


    def call(self, features, training):
        x = self.dense[0](features)
        
        x = self.dense[1](x)

        for i in range(self.N):
            x = self.dense[2 + i](x) + x

        x = self.dense[2 + self.N](x)
        
        x = self.dropout[0](x, training=training)

        return x
