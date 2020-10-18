# import libraries
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformations.slicing import slice_images
from transformations.augmentation import augment_images
from transformations.fixed import get_middle_slices
from transformations.interpolation import create_interpolated_images


class MRDataset(Dataset):


    def __init__(self, task, plane, train_val_test='train', test_paths=None, cut='vertical', 
                augment=None, augment_prob=None, weights=None):
        
        print('[DATALOADER] __init__ task: {}, plane: {}, train: {}'.format(
            task, plane, train_val_test))

        super().__init__()

        # TASK & PLANE
        self.task = task
        self.plane = plane

        # TRAIN / VAL / TEST
        self.train = train_val_test

        # DATA SLICING
        self.cut = cut

        # DATA AUGMENTATION
        self.augment = augment
        self.augment_prob = augment_prob

        if self.train in ['train', 'valid']:
            
            # FOLDER
            self.folder_path = 'MRNet-v1.0/{}/{}/'.format(self.train, plane)
            self.records = pd.read_csv(
                'MRNet-v1.0/{}-{}.csv'.format(self.train, task), 
                header=None, names=['id', 'label'])

            # PATHS AND LABELS
            self.records['id'] = self.records['id'].map( lambda i: '0' * (4 - len(str(i))) + str(i) )
            self.paths = [ self.folder_path + filename + '.npy' for filename in self.records['id'].tolist()]
            self.labels = self.records['label'].tolist()

            # WEIGHTS
            if weights is None:
                pos = np.sum(self.labels)
                neg = len(self.labels) - pos
                self.weights = [1, neg / pos]
            else:
                self.weights = weights

            print('[DATALOADER] __init__ weights: {}'.format(self.weights))

        elif self.train == 'test':

            # PATHS
            self.paths = test_paths

            # NO LABELS OR WEIGHTS
            self.labels = None
            self.weights = None

        else:

            raise Exception('{} not recognized'.format(self.train))


    def __len__(self):
        
        return len(self.paths)


    def __getitem__(self, index):

        # print('[DATALOADER] __getitem__ index: {}'.format(index))

        # Slices for a patient
        array = np.load(self.paths[index])

        # print('[DATALOADER] 1. Shape of the array before any transformation: {}'.format(array.shape))

        # How to cut the slices of the cube
        array = slice_images(array, self.cut)

        # print('[DATALOADER] 2. Shape of the array after slicing: {}'.format(array.shape))

        # Interpolate: Standarize the number of slices
        array = create_interpolated_images(array)

        # print('[DATALOADER] 3. Shape of the array after interpolation: {}'.format(array.shape))

        # Get a fixed number of slices
        array = get_middle_slices(array)

        # print('[DATALOADER] 4. Shape of the array after getting middle slices: {}'.format(array.shape))

        # Transformations
        array = augment_images(array, self.train, self.augment, self.augment_prob, self.plane)

        # print('[DATALOADER] 5. Shape of the array after augmentation: {}'.format(array.shape))

        # print('Shape of the array after transformations: {}'.format(array.shape))

        # labels
        label = self.labels[index]
        label = torch.FloatTensor([label])

        # weights
        if label.item() == 1:
            weight = np.array([self.weights[1]])
            weight = torch.FloatTensor(weight)
        else:
            weight = np.array([self.weights[0]])
            weight = torch.FloatTensor(weight)

        return index, array, label, weight
