# import libraries
import argparse
import torch
import os
import numpy as np
from timeit import default_timer
import seed
import config
from utils import get_model, load_bundle_model
from models import choose
from joblib import load
# https://github.com/aleju/imgaug/issues/537
import numpy
numpy.random.bit_generator = numpy.random._bit_generator


def parse_arguments():
    """ Parse arguments """

    parser = argparse.ArgumentParser()
    
    # input-data-csv-filename that contains lines in the following format:
    # MRNet-v1.0/{valid,test}/{axial,coronal,sagittal}/{4-digit id}.npy
    parser.add_argument('-i', '--input', type=str, default='valid-paths.csv')
    
    # output-prediction-csv-path
    parser.add_argument('-o', '--output', type=str, default='predictions.csv')

    args = parser.parse_args()
    return args


def run(args):
    """ Run the main program """

    ## START

    start = default_timer()

    # DEVICE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # INPUT

    patients = read_input(args.input)

    # Data

    print('[PREDICT] Number of patients: {}'.format(len(patients)))
    print('[PREDICT] Approach: {}'.format(config.APPROACH)) 
    print('[PREDICT] Getting models from: {}'.format(config.MODELS_TO_SUBMIT_APPROACH))

    # PREDICTIONS

    predictions = {}

    for task in config.TASKS:

        print('[PREDICT] Task: "{}"'.format(task))

        # Get predictions
        predictions[task] = get_predictions(patients, task, device)

    # OUTPUT

    P = np.zeros((len(predictions['acl']), 3))
    P[:, 0] = predictions['acl']
    P[:, 1] = predictions['meniscus']
    P[:, 2] = predictions['abnormal']

    # Save array as CSV
    print('[PREDICT] Save predictions at {}'.format(args.output))
    np.savetxt(args.output, P, delimiter=',')

    ## END

    end = default_timer() - start
    minutes, seconds = divmod(end, 60)
    _statement = '[PREDICT] Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds)
    print(_statement)


def read_input(filename):

    # Get all paths
    with open(filename) as f:
        paths = [ line.strip() for line in f.readlines() ]

    # FOR EACH patient
    # MRNet-v1.0/valid/sagittal/1130.npy
    # MRNet-v1.0/valid/coronal/1130.npy
    # MRNet-v1.0/valid/axial/1130.npy

    # Add path to dictionary per plane
    patients = {}
    for path in paths:
        
        # Plane
        plane = path.split('/')[-2]
        # _id = int(path.split('/')[-1].split('.npy')[0])
        patients.setdefault(plane, [])
        patients[plane].append(path)
        
    return patients


def get_predictions(patients, task, device):

    print('[PREDICT] Getting predictions for task "{}"'.format(task))

    preds = {}

    for plane in config.PLANES:

        cut = 'vertical'

        # GET BEST MODEL

        model_data = get_model(task, plane, cut, config.MODELS_TO_SUBMIT_APPROACH)

        # LOAD DATA

        patients_paths = patients[plane]
        
        # PREDICTIONS

        preds[plane] = extract_predictions(task, plane, cut, patients_paths, model_data, device)

    print('[PREDICT] Combining those predictions with a Logistic Regression model')

    # COMBINE PLANES

    X = np.zeros((len(preds['axial']), 3))
    X[:, 0] = preds['axial']
    X[:, 1] = preds['coronal']
    X[:, 2] = preds['sagittal']

    # LOAD MODEL
  
    model_data = load_bundle_model(task, config.MODELS_TO_SUBMIT_APPROACH)
    clf = load(model_data['model_path'])
    y_pred = clf.predict_proba(X)[:, 1] # Logistic Regression

    return np.array(y_pred)


def extract_predictions(task, plane, cut, paths, model_data, device='cpu'):
    """ Extract predictions """
    
    print('[PREDICT] Extracting predictions for task "{}" and plane "{}"'.format(task, plane))

    # Load model on GPU
    mrnet = choose.get_model(model_data['model_architecture'])
    # Load weights
    mrnet.load_state_dict(torch.load(model_data['model_path']))
    mrnet.to(device)

    # Pytorch eval mode
    _ = mrnet.eval()

    # Dataset

    if config.APPROACH == 'pretrained':
        # Pre-trained 
        from loaders.loader_baseline import MRDataset
    elif config.APPROACH == 'slices':
        # Slices
        from loaders.loader_slices import MRDataset

    dataset = MRDataset(
        task, plane, 
        train_val_test='test',
        test_paths=paths,
        cut=cut, 
        augment=model_data['model_augment'], 
        augment_prob=model_data['augment_probability'])
    
    # Data loader
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False)

    # Get predictions
    predictions = []

    with torch.no_grad():

        for index, image, _, _ in loader:
            
            # Copy to CUDA device
            if torch.cuda.is_available():
                image = image.cuda()
            
            # Get prediction
            logit = mrnet(image)
            prediction = torch.sigmoid(logit)
            predictions.append(prediction.item())

    return predictions


if __name__ == "__main__":
    args = parse_arguments()
    run(args)