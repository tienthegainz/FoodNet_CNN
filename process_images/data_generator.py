from scipy.misc import imresize, imread
import numpy as np
from keras.utils import Sequence
from os import listdir
import random
from keras.utils import np_utils
from .img_augmentation import *

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataGenerator(Sequence):

    def __init__(self, data_path, n_classes = 11,
                shuffle=True, batch_size=50):
        'Initialization'
        self.data_path = data_path
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.x = listdir(data_path)
        random.shuffle(self.x) # Shuffle data
        self.y = [img.split('_')[0] for img in self.x]
        self.batch_size=batch_size
    # Parameter for step per epoch
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = np.array([imresize(imread(self.data_path+file_name), (200, 200))
                        for file_name in batch_x])
        """Data has shape of 200, 200, 3"""
        augmentator1 = lambda x: augmentation_img(self.data_path+x, 1)
        data_1 = np.array(map(augmentator1, batch_x))
        data = np.concatenate(data, data_1)
        batch_y += batch_y

        augmentator4 = lambda x: augmentation_img(self.data_path+x, 4)
        data_4 = np.array(map(augmentator4, batch_x))
        data = np.concatenate(data, data_4)
        batch_y += batch_y

        augmentator5 = lambda x: augmentation_img(self.data_path+x, 5)
        data_5 = np.array(map(augmentator5, batch_x))
        data = np.concatenate(data, data_5)
        batch_y += batch_y

        data = data/255.00
        labels = np_utils.to_categorical(np.array(batch_y), num_classes=self.n_classes)
        return data, labels
