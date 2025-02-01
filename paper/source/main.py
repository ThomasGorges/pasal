import os
import constants
from processing.dataset import Dataset
from model.universal_scent import UniversalScent
from datetime import datetime
import numpy as np
from utility import log, tf_util, score, parameters
from utility.study import Study
import argparse
import utility.parameters
import sys
import subprocess
import json
import logging
from time import sleep
import pandas as pd


def create_model_with_config(summary_path_with_run, params, steps_per_epoch, blacklisted_dbs=None, alpha=None, beta=None):
    parameters.reinit_seed(params['Seed'])

    if alpha:
        params['Alpha'] = float(alpha)
    if beta:
        params['Beta'] = float(beta)

    log.write_config(os.path.join(summary_path_with_run, 'config.json'), params)

    model = UniversalScent(summary_path_with_run, params, steps_per_epoch, blacklisted_dbs)

    return model

def objective(fold_id, params: dict, summary_path: str, silent=True, exclude_list=None):
    parameters.reinit_seed(params['Seed'])

    result_folder = os.path.join(summary_path, 'results/')
    os.mkdir(result_folder)

    config_file_path = os.path.join(summary_path, 'config.json')
    with open(config_file_path, 'w') as f:
        json.dump(params, f)

    # If not finished, we start a new worker to train
    flags = ['--retrain', config_file_path, '--outputpath', summary_path]
        
    if silent:
        flags += ['--silent']

    if exclude_list:
        flags += ["--exclude", exclude_list]
    
    flags += ['--fold_id', str(fold_id)]

    subprocess.Popen(["python3", "main.py", *flags])

    # Wait until worker finishes
    while not os.path.exists(summary_path + '/results.json'):
        sleep(3)

    # Read avg validation loss and Z-score
    with open(summary_path + '/results.json', 'r') as f:
        sleep(10)
        run_results = json.load(f)

    return run_results


def train_with_config(summary_path, params, is_silent, fold_id, exclude_list=[], alpha=None, beta=None, use_embeddings=False):
    batch_size = int(params['Batchsize'])

    logging.info('Config: ' + str(params))

    if not os.path.exists(os.path.join(summary_path, 'checkpoints')):
        os.makedirs(os.path.join(summary_path, 'checkpoints'))
    
    
    datasets = {
        'train': Dataset(db_type='train', fold_id=fold_id, exclude_list=exclude_list, use_embeddings=use_embeddings),
        'val': Dataset(db_type='val', fold_id=fold_id, use_embeddings=use_embeddings),
        'test': Dataset(db_type='test', use_embeddings=use_embeddings),
    }

    datasets['train'].load(batch_size, shuffle=True)
    datasets['val'].load(-1, shuffle=False)
    datasets['test'].load(-1, shuffle=False)

    steps_per_epoch = datasets['train'].steps_per_epoch

    model = create_model_with_config(summary_path, params, steps_per_epoch, blacklisted_dbs=exclude_list, alpha=alpha, beta=beta)

    dream_val_score = model.train(datasets, is_silent)

    # Save model
    store_as_checkpoint(model, summary_path)

    log.export_train_history(summary_path, model.loss_history, model.accuracy_history, model.loss_weighting.get_history())

    log.export_results(summary_path, dream_val_score)

    z_score = float(dream_val_score['z_score'])
    mse_loss = float(dream_val_score['mse_loss'])

    if not is_silent:
        print(f'Validation MSE-loss: {mse_loss} {summary_path}')
        print(f'Validation Z-Score: {z_score} {summary_path}')

    return mse_loss


def store_as_checkpoint(model, path):
    import tensorflow as tf

    checkpoint_path = path + 'checkpoints/'
    
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=1)

    checkpoint_manager.save()


def restore_from_checkpoint(path):
    import tensorflow as tf

    # Restore checkpoint
    checkpoint_path = path + 'checkpoints/'
    
    config_path = os.path.join(path, 'config.json')
    params = utility.parameters.restore_from_file(config_path)

    model = UniversalScent('', params, 1, None)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=1)
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)

    # Partial is expected because not all weights need to be used (IFRA, Leffingwell, optimizer, ...)
    # With this function, warnings are suppressed
    status.expect_partial()

    return model


def predict_dream_testset(model, dataset):
    dream_predictions = model.predict_dream_testset(dataset)
    scores = score.calculate_scores_from_submission_in_memory(constants.GS_CID_ORDER, constants.GS_GROUND_TRUTH, dream_predictions)

    z_score = float(scores['z_score'])
    print(f'DREAM test-set Z-Score: {z_score}')


def export_shared_model_features(path, model):
    dataset = Dataset(db_type='train')
    dataset.load(batch_size=64, shuffle=False)

    features = model.predict_features([dataset])
    features.to_csv(path + '/features.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniversalScent to predict odors based on SMILES')
    parser.add_argument('--restore', dest='checkpoint_path', help='Restore trained model')
    parser.add_argument('--outputpath', dest='output_path', help='Path to write train results')
    parser.add_argument('--calc_z_score', dest='calc_z_score', help='Path to a submission file in DREAM format')
    parser.add_argument('--retrain', dest='retrain_config', help='Path to a config file storing train parameters')
    parser.add_argument('--silent', dest='silent', help='Try to make no output to stdout', action='store_true')
    parser.add_argument('--num_workers', dest='num_workers', help='Number of workers if run parallel is set')
    parser.add_argument('--use_val_for_training', dest='use_val_for_training', action='store_true', help='Use validation set as additional training data. Useful after finding the optimal hyperparameters on the validation set')
    parser.add_argument('--export_features', dest='export_features', action='store_true', help='Flag to export features from the shared model')
    parser.add_argument('--random_search_seed', dest='random_search_seed')
    parser.add_argument('--num_models', dest='num_models')
    parser.add_argument('--study_name', dest='study_name')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument("--fetch_results", action="store_true")
    parser.add_argument('--fold_id', dest='fold_id')
    parser.add_argument("--exclude", dest="exclude_list")
    parser.add_argument('--alpha', dest='alpha')
    parser.add_argument('--beta', dest='beta')
    parser.add_argument('--use_embeddings', dest='use_embeddings', action='store_true')

    args = parser.parse_args()

    if args.calc_z_score:
        z_score = score.calculate_scores_from_submission(args.calc_z_score)['z_score']
        print(f'Z-Score: {z_score}')
        sys.exit(0)

    tf_util.enable_deterministic_behaviour()
    tf_util.allow_dynamic_memory_growth()

    num_workers = 1
    if args.num_workers:
        if args.num_workers == 'auto':
            num_workers = -1
        else:
            num_workers = int(args.num_workers)

    timestamp = datetime.now().isoformat()

    if args.fetch_results:
        fold_ids_to_fetch = []

        if args.fold_id:
            fold_ids_to_fetch = [args.fold_id]
        else:
            fold_ids_to_fetch = np.arange(constants.NUM_FOLDS, dtype=np.int32).tolist()

        for fold_id in fold_ids_to_fetch:
            study = Study(name=args.study_name + '_' + str(fold_id))
            study.create(resume_if_exists=True)
            study.export_results()
    elif args.checkpoint_path and args.export_features and not args.retrain_config:
        model = restore_from_checkpoint(args.checkpoint_path)
        
        export_shared_model_features(args.checkpoint_path, model)
    elif args.retrain_config and args.ensemble:
        if 'study:///' not in args.retrain_config:
            raise Exception('Study name needed')
        
        study_name = args.retrain_config.split('/')[-1]
        df = pd.read_csv('study_results/' + study_name + '_best_models.csv')
        df = df.set_index('idx')

        pids = []

        for run_id in df.index.values:
            fold_id = str(int(df.loc[run_id]['fold_id']))
            orig_idx = str(int(df.loc[run_id]['orig_idx']))
            flags = ['--retrain', f'study:///{study_name}_{fold_id}/{orig_idx}', '--fold_id', fold_id]#, '--silent']

            if args.use_val_for_training:
                flags += ['--use_val_for_training']
            
            pid = subprocess.Popen(["python3", "main.py", *flags])

            pids.append(pid)
        
        print('Waiting for training to finish...')
        for pid in pids:
            pid.wait()
            
    elif args.retrain_config and not args.ensemble:
        params = utility.parameters.restore_from_file(args.retrain_config)
        if args.output_path:
            output_path = args.output_path
        else:
            if 'study:///' in args.retrain_config:
                study_name = args.retrain_config.split('/')[-2]
                config_id = args.retrain_config.split('/')[-1]

                output_path = '../output/study_results/' + study_name

                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                
                output_path += '/' + config_id

                if args.alpha:
                    output_path += f'_a{float(args.alpha):.2f}'
                if args.beta:
                    output_path += f'_b{float(args.beta):.2f}'

                output_path += '/'
            else:
                output_path = 'temp/'

        logging.info(f'Training with params ({params}). Output will be written to {output_path}.')

        train_with_config(output_path, params, args.silent, args.fold_id, exclude_list=args.exclude_list, alpha=args.alpha, beta=args.beta, use_embeddings=args.use_embeddings)
    else:
        if args.random_search_seed:
            seed = int(args.random_search_seed)
        else:
            seed = constants.SEED
        
        with open('hyperparameters.json') as f:
            search_space = json.load(f)
        
        if args.study_name:
            study_name = args.study_name

            if args.fold_id:
                study_name += '_' + str(args.fold_id)
        else:
            study_name = ''
        
        study = Study(name=study_name)
        study.create_or_resume()
        study.optimize(objective, args.fold_id, search_space, num_trials=int(args.num_models), num_parallel_jobs=int(num_workers), exclude_list=args.exclude_list)
