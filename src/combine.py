# import libraries
import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import seed
import config
from joblib import dump
from utils import get_model
from models import choose
from timeit import default_timer
from sms import send
from datetime import datetime
import csv
import pandas as pd
import shutil
# https://github.com/aleju/imgaug/issues/537
import numpy
numpy.random.bit_generator = numpy.random._bit_generator


def parse_arguments():
    """ Parse arguments """

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=config.TASKS)
    args = parser.parse_args()
    return args


def run(args):
    """ Run the main program """

    ## START

    start = default_timer()

    ## TASK

    task = args.task
    print('[COMBINE] Task: "{}"'.format(task))

    # DEVICE

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[COMBINE] Device: {}'.format(device))

    # TRAIN

    results = {}
    model_details = {}

    for plane in config.PLANES:

        for cut in config.SLICING:

            print('[COMBINE] Trainig data. Plane: "{}" Slice "{}"'.format(plane, cut))

            # Get model
            model_data = get_model(task, plane, cut, config.TRAIN_MODELS_PATH_APPROACH)

            # Copy model
            best_model_stored_for_submission = \
                get_model(task, plane, cut, config.MODELS_TO_SUBMIT_APPROACH)
            if best_model_stored_for_submission is None or \
                model_data['model_val_auc'] > best_model_stored_for_submission['model_val_auc']:
                # Save model for submission
                import shutil
                print('Copying model...!')
                destination = os.path.join(config.MODELS_TO_SUBMIT_APPROACH, model_data['model_name'])
                shutil.copyfile(model_data['model_path'], destination)

            # Save them
            model_details[(plane, cut)] = model_data

            # Get Predictions
            predictions, labels = extract_predictions(
                task, plane, cut,
                model_data,
                train_val_test='train', 
                shuffle=False,
                device=device)

            # Save
            results[plane + '-' + cut] = predictions
            # results['labels'] = labels

    # Concatenate predictions
    print('[COMBINE] Training data: concatenating features for the planes')
    X = np.zeros((len(predictions), len(config.PLANES) * len(config.SLICING)))
    # X[:, 0] = results['axial-vertical']
    # X[:, 1] = results['coronal-vertical']
    # X[:, 2] = results['sagittal-vertical']
    i = 0
    for plane in config.PLANES:
        for cut in config.SLICING:
            X[:, i] = results[plane + '-' + cut]
            i += 1

    # Add labels
    y = np.array(labels)

    # Fit a model
    print('[COMBINE] Fitting a Logistic Regression model')
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)

    ## EVALUATE 

    results_val = {}

    for plane in config.PLANES:

        for cut in config.SLICING:

            # Get model
            model_data = model_details[(plane, cut)]
            
            # Get predictions
            predictions, labels = extract_predictions(
                task, plane, cut,
                model_data,
                train_val_test='valid', 
                shuffle=False,
                device=device)
            
            # Save
            results_val[plane + '-' + cut] = predictions
            # results_val['labels'] = labels

    # Concatenate predictions
    print('[COMBINE] Evaluating data: concatenating features for the planes')
    X_val = np.zeros((len(predictions), len(config.PLANES) * len(config.SLICING)))
    # X_val[:, 0] = results_val['axial-vertical']
    # X_val[:, 1] = results_val['coronal-vertical']
    # X_val[:, 2] = results_val['sagittal-vertical']
    i = 0
    for plane in config.PLANES:
        for cut in config.SLICING: 
            X_val[:, i] = results_val[plane + '-' + cut]
            i += 1

    # Add labels
    y_val = np.array(labels)

    # Get prob predictions
    y_pred = clf.predict_proba(X_val)[:, 1] # Logistic Regression
    # y_pred = clf.predict(X_val) # SVM / Bayesian Ridge

    # LR Coefficients
    lr_statement = '[COMBINE] Logistic Regression coefficients: {}'.format(
        ', '.join([ '{:.4f}'.format(c) for c in clf.coef_[0] ]))
    print(lr_statement)

    # Get metric
    auc = roc_auc_score(y_val, y_pred)
    auc_statement = '[COMBINE] ## Val AUC for task "{}" is: {:.4f} ##'.format(task, auc)
    print(auc_statement)

    # Print models' details
    model_statements = ''
    for key, value in model_details.items():
        _statement = 'Model for "{}": {}'.format(key, value)
        model_statements +=  _statement + '\n'

    # Save model

    clf_model_name = 'clf_{}_val_auc_{:.4f}.joblib'.format(task, auc)
    clf_model_path = '{}/{}'.format(config.TRAIN_MODELS_PATH_APPROACH, clf_model_name)
    dump(clf, clf_model_path)
    print('[COMBINE] Model "{}" saved'.format(clf_model_path))
    import shutil
    print('Copying model...!')
    destination = os.path.join(model_to_submit_path, clf_model_name)
    shutil.copyfile(clf_model_path, destination)

    ## END

    end = default_timer() - start
    minutes, seconds = divmod(end, 60)
    exec_statement = 'Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds)
    print(exec_statement)

    ## SEND TEXT

    _statement = auc_statement + '\n' + model_statements + exec_statement
    send(_statement)


def extract_predictions(task, plane, cut, model_data, train_val_test, shuffle=False, device='cpu'):
    """ Extract predictions """

    print('[COMBINE] Extracting predictions for task "{}" and plane "{}"'.format(task, plane))

    # Save predictions in a file for easier retrieval

    model_filename = model_data['model_path'].split(os.path.sep)[-1] + '-' + train_val_test
    predictions_filename = os.path.join(
        config.PREDICTIONS_PATH_APPROACH, model_filename + '.csv')
    
    if os.path.exists(predictions_filename):

        # read 
        df = pd.read_csv(predictions_filename)
        
        # Get predictions and labels
        labels = df['label'].values
        predictions = df['prediction'].values

    else:

        # Load model on GPU
        mrnet = choose.get_model(model_data['model_architecture'])
        mrnet.load_state_dict(torch.load(model_data['model_path']))
        mrnet.to(device)

        # Pytorch eval mode

        _ = mrnet.eval()

        # Parameters
        augment = model_data['model_augment']
        prob = model_data['augment_probability']

        # Instantiate the dataset and loader

        if config.APPROACH == 'pretrained':
            # Pre-trained 
            from loaders.loader_baseline import MRDataset
        elif config.APPROACH == 'slices':
            # Slices
            from loaders.loader_slices import MRDataset

        dataset = MRDataset(task, plane, train_val_test=train_val_test, cut=cut, 
            augment=augment, augment_prob=prob)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=shuffle, drop_last=False)

        # Get predictions

        predictions = []
        labels = []
        rows = []

        with torch.no_grad():

            for index, image, label, weight in tqdm(loader):
                
                # Copy to CUDA device
                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda()
                
                # Get prediction
                logit = mrnet(image.float())
                prediction = torch.sigmoid(logit).item()
                predictions.append(prediction)
                # Get label
                label = label.item()
                labels.append(label)
                # Append row
                index = index.item()
                rows.append( [ index, prediction, label] )

        # Write predictions and labels locally

        with open(predictions_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            header = ['index', 'prediction', 'label']
            writer.writerow(header)
            writer.writerows(rows)

    return predictions, labels


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
