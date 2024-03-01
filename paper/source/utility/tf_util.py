import tensorflow as tf


def enable_deterministic_behaviour():
    tf.config.threading.set_inter_op_parallelism_threads = 1
    tf.config.threading.set_intra_op_parallelism_threads = 1


def allow_dynamic_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
