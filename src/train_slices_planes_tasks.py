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

# HYPERPARAMETERS


def parse_arguments():
    """ Parse arguments """

    parser = argparse.ArgumentParser()

    # task
    # parser.add_argument('-t', '--task', type=str, required=True,
    #                     choices=['abnormal', 'acl', 'meniscus'])

    # experiment's prefix name
    parser.add_argument('--prefix_name', type=str, required=True)

    # model
    parser.add_argument('--model', type=str, default='resnet18')

    # epochs
    parser.add_argument('--epochs', type=int, default=500)

    # Image Slicing
    # Way slices are made (new planes are created this way)
    # vertical (default), horizontal, diagonal
    parser.add_argument('--cut', type=str, default='vertical')

    # augmenting images
    parser.add_argument('--augment', type=str, default='albumentations-group')

    # augmenting probability
    parser.add_argument('--augment_prob', type=float, default=0.00)

    # # patience
    # parser.add_argument('--patience', type=int, default=5)

    # # LR scheduler
    # parser.add_argument('--lr_scheduler', type=str,
    #                 choices=['plateau', 'step'], default='plateau')

    # # gamma
    # parser.add_argument('--gamma', type=float, default=0.5)

    # remove history
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)

    # save model
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)

    # logging
    parser.add_argument('--log_every', type=int, default=100)

    args = parser.parse_args()
    return args


def run(args):
    """ Run training """

    ## START

    start = default_timer()

    # DEVICE

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LOGS

    # Create a folder that'll be used by tensorboard 
    # to save the training logs and visualize the metrics of the training session
    # On each run of the script, a new folder (named after the timestamp) is created:
    
    # log_root_folder = "./logs/{}/".format(args.task)
    log_root_folder = "./logs/"

    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = os.path.join(log_root_folder, now.strftime("%Y%m%d-%H%M%S"))

    print('[TRAIN] Creating logs folder: "{}"'.format(logdir))

    # Create dir
    os.makedirs(logdir)

    MAIN_LOG = os.path.join(logdir, 'mainlog.txt')

    with open(MAIN_LOG, 'w') as f:
        print("""MRNET 
Experiment: {}
Prefix: {}
Model: {}
Save Model: {}
Device: {}
Epochs: {}
Augmentation: {}
LR: {}
lr_scheduler: {}
patience: {}""".format(
    now.strftime("%d%m%Y-%H%M%S"), args.prefix_name, args.model, args.save_model, 
    device, args.epochs, args.augment, config.LEARNING_RATE, '', ''), file=f)

    # args.lr_scheduler
    # args.patience

    ## TENSORBOARD

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(logdir)

    ## DATA LOADER

    print('[TRAIN] Loading Data Loaders')

    from loaders.lastloader import MRDataset

    ## AUGMENTATION

    augment = args.augment
    augment_prob = args.augment_prob

    ## CUT

    cut = args.cut
    
    # Instantiate a train and validation MRDataset(s)

    # args.task, 
    train_dataset = MRDataset(train_val_test='train', cut=cut, augment=augment, augment_prob=augment_prob)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, drop_last=False)

    validation_dataset = MRDataset(train_val_test='valid', cut=cut, augment=augment, augment_prob=augment_prob)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, drop_last=False)

    # MODEL

    print('[TRAIN] Instantiate or loading model: {}'.format(args.model))

    from models import choose

    mrnet = choose.get_model(args.model)
    mrnet.to(device)

    model_name = args.model
    if 'arch_' in model_name:
        model_name = model_name.split('arch_')[1].split('_')[0]

    # OPTIMIZER

    print('[TRAIN] Defining Optimizer')

    # Adam optimizer as well as a learning rate scheduler

    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.1

    optimizer = optim.Adam(
        mrnet.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY)
    
    # optimizer = optim.SGD(
    #    mrnet.parameters(), 
    #    lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=3, 
        factor=.3, 
        threshold=1e-4, 
        verbose=True)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    ## TRAIN & EVALUATE

    print('[TRAIN] Starting Training!')

    num_epochs = args.epochs
    iteration_change_loss = 0
    # patience = args.patience
    log_every = args.log_every

    print('[TRAIN] Epochs = {}'.format(num_epochs))

    t_start_training = time.time()

    for epoch in range(num_epochs):

        print('[TRAIN] EPOCH # {}'.format(epoch + 1))

        # Learning Rate

        current_lr = get_lr(optimizer)

        # Start timer

        t_start = time.time()

        # TRAIN

        train_loss, train_auc = train_model(
            mrnet, train_loader, device, epoch, num_epochs, optimizer, 
            writer, MAIN_LOG, current_lr, log_every)
        
        # EVALUATE

        val_loss, val_auc, val_aucs = evaluate_model(
            mrnet, validation_loader, device, epoch, num_epochs, 
            writer, MAIN_LOG, current_lr, log_every // 5)

        # Scheduler

        # if args.lr_scheduler == 'plateau':
        #     scheduler.step(val_loss)
        # elif args.lr_scheduler == 'step':
        #     scheduler.step()

        # End timer

        t_end = time.time()
        delta = (t_end - t_start) / 60.

        # Print
        _statement = """Epoch {}:\nTrain loss: {:.4f} | Train AUC: {:.4f}\nVal Loss: {:.4f} | Val AUC: {:.4f}\nElapsed time: {:.2f} min""".format(
            epoch + 1, train_loss, train_auc, val_loss, val_auc, delta)
        if val_auc > best_val_auc: _statement += '\nModel will be saved'
        with open(MAIN_LOG, 'a') as f: print(_statement, file=f)
        # Send
        with open(MAIN_LOG) as f:
            text = f.read()
        send(text)

        # LEARNING RATE Update?

        # if args.lr_scheduler == 1:
        #     print('[RUN][INFO] Updating learning rate..')
        #     scheduler.step(val_loss)

        iteration_change_loss += 1
        print('-' * 100)

        # SAVE MODEL

        if val_auc > best_val_auc:
            # Update best
            best_val_auc = val_auc
            if bool(args.save_model):
                # # Remove previous models
                # for f in os.listdir('./models/'):
                #     if args.prefix_name in f: # args.task in f and 
                #         os.remove('./models/{}'.format(f))
                # Save
                model_filename = 'models/model_{}_train_auc_{:.4f}_val_auc_{:.4f}_{:.4f}_{:.4f}_{:.4f}_train_loss_{:.4f}_val_loss_{:.4f}_epoch_{}_arch_{}_cut_{}_augment_{}_augment-probability_{}.pth'.format(
                    args.prefix_name, train_auc, val_auc, val_aucs[0], val_aucs[1], val_aucs[2], 
                    train_loss, val_loss, epoch + 1, 
                    model_name, cut, augment, augment_prob)
                torch.save(mrnet.state_dict(), model_filename)
                _statement = 'Model saved: "{}"'.format(model_filename)
                print(_statement)

        # PATIENCE

        # if val_loss < best_val_loss:
        #     # Save best val loss
        #     best_val_loss = val_loss
        #     iteration_change_loss = 0

        # if iteration_change_loss == patience:
        #     _statement = 'Early stopping after {0} iterations without the decrease of the val loss'.format(iteration_change_loss)
        #     print(_statement)
        #     with open(MAIN_LOG, 'a') as f: print(_statement, file=f)
        #     # Send
        #     with open(MAIN_LOG) as f:
        #         text = f.read()
        #     send(text)
        #     break

    ## END

    end = default_timer() - start
    minutes, seconds = divmod(end, 60)
    _statement = '[INFO] Execution duration: {:.2f} minutes {:.2f} seconds'.format(minutes, seconds)
    print(_statement)
    with open(MAIN_LOG, 'a') as f: print(_statement, file=f)

    ## SEND TEXT

    with open(MAIN_LOG) as f:
        text = f.read()
    send(text)


def train_model(model, train_loader, device, epoch, num_epochs, optimizer, writer, log_file, current_lr, log_every=100):
    
    print('[TRAIN] Train model')

    # Train mode
    _ = model.train()

    # Clean up
    torch.cuda.empty_cache()

    # Init y_preds and y_trues
    y_preds = {}
    y_trues = {}
    for k in [0, 1, 2]:
        y_preds.setdefault(k, [])
        y_trues.setdefault(k, [])

    # Init losses
    losses = []

    # Number of steps
    total_step = len(train_loader)

    for i, (image, label, weight) in enumerate(train_loader):

        # print(i)
        # print(image.shape)
        # print('Memory allocated: {:.2f} MiB'.format(torch.cuda.memory_allocated() / 1.049e6))
        # print('Memory cached: {:.2f} MiB'.format(torch.cuda.memory_cached() / 1.049e6))
        
        # clear out the gradients
        optimizer.zero_grad()

        # Copy to CUDA device
        image = image.to(device)
        label = label.to(device)
        weight = weight.to(device)

        # Forward pass
        prediction = model.forward(image.float())

        # Values
        labels = label[0]
        weights = weight[0]
        predictions = prediction[0]

        # Loss
        loss = torch.nn.BCEWithLogitsLoss(weight=weights)(predictions, labels)

        # Backward and Optimize
        loss.backward()
        optimizer.step()

        # Prediction
        pred_logits = torch.sigmoid(prediction)
        preds = pred_logits[0]

        # Append labels and predictions
        for k in range(3):
            y_preds[k].append(preds[k].item())
            y_trues[k].append(labels[k].item())

        # Evaluation: AUC
        aucs = [ evaluate_auc(y_trues[k], y_preds[k]) for k in range(3) ]
        auc = np.mean(aucs)

        # Loss
        loss_value = loss.item()
        losses.append(loss_value)

        # print('{}. Train Loss: {:.4f},  Train AUC: {:.4f}, Label: {}, Weight: {:.4f}, Prediction: {:.4f}'.format(
        #     i + 1, loss_value, auc, y_true, weight.item(), y_pred))

        # Tensorboard
        if writer:
            writer.add_scalar('Train/Loss', loss_value, epoch * len(train_loader) + i)
            writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        # Print every N
        if i > 0 and i % log_every == 0:
            _statement = """Epoch [{0} / {1}] Step: [{2} / {3}] avg_train_loss: {4:.4f} | 
avg_train_auc: {5:.4f} [ACL: {6:.2f}, Meniscus {7:.2f}, Abnormal: {8:.2f}] | lr : {9}""".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    total_step,
                    np.mean(losses),
                    auc,
                    aucs[0],
                    aucs[1],
                    aucs[2],
                    current_lr
                )
            print(_statement)

        # print('Memory allocated: {:.2f} MiB'.format(torch.cuda.memory_allocated() / 1.049e6))
        # print('Memory cached: {:.2f} MiB'.format(torch.cuda.memory_cached() / 1.049e6))

        # Clean up
        del image
        del label
        del weight
        torch.cuda.empty_cache()

    if writer: 
        writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    # Round metrics
    train_loss_epoch = np.mean(losses)
    train_auc_epoch = auc

    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, device, epoch, num_epochs, writer, log_file, current_lr, log_every=20):
    
    print('[EVALUATE] Evaluate model')

    _ = model.eval()

    # Init y_preds and y_trues
    y_preds = {}
    y_trues = {}
    for k in [0, 1, 2]:
        y_preds.setdefault(k, [])
        y_trues.setdefault(k, [])

    # Init losses
    losses = []

    with torch.no_grad():

        for i, (image, label, weight) in enumerate(val_loader):

            # Copy to CUDA device
            image = image.to(device)
            label = label.to(device)
            weight = weight.to(device)
        
            # Label, weight
            labels = label[0]
            weights = weight[0]

            # Prediction
            predictions = model.forward(image.float())[0]

            # Loss
            loss = torch.nn.BCEWithLogitsLoss(weight=weights)(predictions, labels)
            loss_value = loss.item()
            losses.append(loss_value)

            # Sigmoid prob
            probabilities = torch.sigmoid(predictions)

            # Append labels and predictions
            for k in range(3):
                y_preds[k].append(probabilities[k].item())
                y_trues[k].append(labels[k].item())

            # Evaluation: AUC
            aucs = [ evaluate_auc(y_trues[k], y_preds[k]) for k in range(3) ]
            auc = np.mean(aucs)

            # Print
            # print('{}. Val Loss: {:.4f},  Val AUC: {:.4f}, Label: {}, Weight: {:.4f}, Prediction: {:.4f}, Sigmoid prob: {:.4f}'.format(
            #     i + 1, loss_value, auc, label[0], weight[0], prediction[0], probability))

            if writer:
                writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
                writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

            # Print every N
            if i > 0 and i % log_every == 0:
                _statement = """[Epoch: {0} / {1} | Step: {2} / {3}] avg_val_loss: {4:.4f} | 
avg_val_auc: {5:.4f} [ACL: {6:.2f}, Meniscus {7:.2f}, Abnormal: {8:.2f}] | lr: {9}""".format(
                        epoch + 1,
                        num_epochs,
                        i,
                        len(val_loader),
                        np.mean(losses),
                        auc,
                        aucs[0],
                        aucs[1],
                        aucs[2],
                        current_lr
                    )
                print(_statement)

            # Clean up
            del image
            del label
            del weight
            torch.cuda.empty_cache()

    _statement = """[Epoch: {0} / {1} | Step: {2} / {3}] avg_val_loss: {4:.4f} | 
avg_val_auc: {5:.4f} [ACL: {6:.2f}, Meniscus {7:.2f}, Abnormal: {8:.2f}] | lr: {9}""".format(
            epoch + 1,
            num_epochs,
            i,
            len(val_loader),
            np.mean(losses),
            auc,
            aucs[0],
            aucs[1],
            aucs[2],
            current_lr
        )
    print(_statement)
    
    if writer: 
        writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.mean(losses)
    val_auc_epoch = auc
    val_aucs_epoch = aucs

    return val_loss_epoch, val_auc_epoch, val_aucs_epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    args = parse_arguments()
    run(args)