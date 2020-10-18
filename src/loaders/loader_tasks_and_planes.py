# import libraries
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
from transformations.slicing import slice_images
from transformations.augmentation import augment_images
from transformations.fixed import get_middle_slices
from transformations.interpolation import create_interpolated_images


class MRDataset(Dataset):

    def __init__(self, train_val_test='train', test_paths=None, cut='vertical', 
                augment=None, augment_prob=None, weights=None):
        
        print('[DATALOADER] __init__ train: {}'.format(train_val_test))

        super().__init__()

        # TRAIN / VAL / TEST
        self.train = train_val_test

        # DATA SLICING
        self.cut = cut

        # DATA AUGMENTATION
        self.augment = augment
        self.augment_prob = augment_prob

        if self.train in ['train', 'valid']:

            self.paths = {}
            self.labels = {}
            self.weights = {}

            # Grab labels for each task

            for task in config.TASKS:

                # RECORDS
                records = pd.read_csv(
                    'MRNet-v1.0/{}-{}.csv'.format(self.train, task), 
                    header=None, names=['id', 'label'])

                # LABELS
                self.labels[task] = records['label'].tolist()

                # WEIGHTS
                pos = np.sum(self.labels[task])
                neg = len(self.labels[task]) - pos
                self.weights[task] = [1, neg / pos]

            # Grab images for each plane

            for plane in config.PLANES:

                # PATHS
                folder_path = 'MRNet-v1.0/{}/{}/'.format(self.train, plane)
                self.paths[plane] = [ '{}/{:04d}.npy'.format(folder_path, filename) 
                    for filename in records['id'].tolist() ]

        elif self.train == 'test':

            # PATHS
            self.paths = test_paths

            # NO LABELS OR WEIGHTS
            self.labels = None
            self.weights = None

        else:

            raise Exception('{} not recognized'.format(self.train))

    def __len__(self):
      
        return len(self.paths['axial'])

    def __getitem__(self, index):

        # print('[DATALOADER] __getitem__ index: {}'.format(index))

        stack = []

        for plane in config.PLANES:

            # Slices for a patient
            slices = np.load(self.paths[plane][index])

            # print('[DATALOADER] 1. Shape of the array before any transformation: {}'.format(slices.shape))

            # How to cut the slices of the cube
            array = slice_images(slices, self.cut)

            # print('[DATALOADER] 2. Shape of the array after slicing: {}'.format(array.shape))

            # Interpolate: Standarize the number of slices
            array = create_interpolated_images(array)

            # print('[DATALOADER] 3. Shape of the array after interpolation: {}'.format(array.shape))

            # Get a fixed number of slices
            array = get_middle_slices(array)

            # print('[DATALOADER] 4. Shape of the array after getting middle slices: {}'.format(array.shape))

            # Transformations
            array = augment_images(array, self.train, self.augment, self.augment_prob, plane)

            # print('[DATALOADER] 5. Shape of the array after augmentation: {}'.format(array.shape))

            stack.append(array)

        # Concatenate arrays!

        # Array of slices for each plane
        stack = np.concatenate(stack)

        # print('Shape of the array after transformations: {}'.format(array.shape))

        # Array of Labels for each task
        labels = torch.FloatTensor([ self.labels[task][index] for task in TASKS ])

        # Array of Weights for each task
        weights =  torch.FloatTensor([ self.weights[task][1] 
            if self.labels[task][index] == 1 else self.weights[task][0]
            for task in config.TASKS ])

        return stack, labels, weights

