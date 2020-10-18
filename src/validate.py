# get validation AUC

# import libraries
import argparse
import shutil
import os
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from sklearn import metrics
from utils import evaluate_auc

from timeit import default_timer
import seed
from sms import send
import config
# import pdb; pdb.set_trace();

# https://github.com/aleju/imgaug/issues/537
import numpy
numpy.random.bit_generator = numpy.random._bit_generator


def parse_arguments():
    """ Parse arguments """

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()
    return args


def run(args):
    """ Run validation """

    ## START

    start = default_timer()

    # DEVICE

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MODEL

    print('[TRAIN] Instantiate or loading model: {}'.format(args.model))

    from models import choose

    mrnet = choose.get_model(args.model)
    mrnet.to(device)

    model_name = args.model.split('arch_')[1].split('_')[0]
    augment = args.model.split('augment_')[1].split('_')[0]
    augment_prob = float(args.model.split('augment-probability_')[1].split('.pth')[0])

    ## DATA LOADER

    print('[TRAIN] Loading Data Loaders')

    from loaders.lastloader import MRDataset

    validation_dataset = MRDataset(train_val_test='valid', cut='vertical', 
        augment=augment, augment_prob=augment_prob)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, drop_last=False)

    # validation

    from supertrain import evaluate_model

    val_loss, val_auc, val_aucs = evaluate_model(
        mrnet, validation_loader, device, epoch=1, num_epochs=1, 
        writer=None, log_file=None, current_lr=None, log_every=20)

    print(val_loss)
    print(val_auc)
    print(val_aucs)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)