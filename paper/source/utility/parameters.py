from pickletools import optimize
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import logging
import random
import optuna


def __generate_new_value(trial: optuna.trial.Trial, search_space, field_name):
    """Generate a new single random training parameter.

    :param field: Dict containing name, type and possible value/range. See hyperparameters.py for example input.
    :return: Dict containing random generated training parameter.
    """

    name = field_name
    args = search_space[name]

    optimize_arg = True

    if 'optimize' in args and args['optimize'] == False:
        optimize_arg = False
    
    generated_value = None

    if args['type'] == 'categorical':
        if optimize_arg:
            generated_value = trial.suggest_categorical(name, choices=args['choices'])
        else:
            generated_value = random.choice(args['choices'])
    elif args['type'] == 'uniform':
        if optimize_arg:
            generated_value = trial.suggest_uniform(name, low=args['low'], high=args['high'])
        else:
            generated_value = random.uniform(args['low'], args['high'])
    elif args['type'] == 'int':
        if optimize_arg:
            generated_value = trial.suggest_int(name, low=args['low'], high=args['high'], step=args['step'])
        else:
            generated_value = random.randrange(args['low'], args['high'] + 1, args['step'])
    elif args['type'] == 'float':
        if optimize_arg:
            generated_value = trial.suggest_float(name, low=args['low'], high=args['high'], step=args['step'])
        else:
            raise NotImplementedError()
    elif args['type'] == 'constant':
        generated_value = args['value']
    else:
        raise Exception
    
    if not optimize_arg or args['type'] == 'constant':
        trial.set_user_attr(name, generated_value)
    
    return generated_value


def generate_new_parameters(trial: optuna.trial.Trial, search_space: dict):
    """Generate new training parameters based on the specification in hyperparameters.py.

    :return: Dict containing all new training parameters.
    """
    new_params = {}

    for param in search_space.keys():
        value = __generate_new_value(trial, search_space, param)

        new_params[param] = value

    return new_params


def restore_from_file(filepath):
    """This function restores the configs from an already trained model.

    :param filepath: A string containing the file path.
    :return: Dict containing training configs.
    """
    if filepath.startswith('study:///'):
        # Load from study instead
        study_name = filepath.split('/')[-2]
        config_id = int(filepath.split('/')[-1])
        
        df = pd.read_json(f'../output/study_results/{study_name}.json', orient='index')

        params = df.loc[config_id].to_dict()

        del params['user_attrs']
    else:
        with open(filepath, 'r') as f:
            params = json.load(f)

    reinit_seed(params['Seed'])

    return params


def generate_new_seed():
    new_seed = random.getrandbits(32)

    return new_seed


def reinit_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    logging.info('Using new seed: ' + str(seed))
