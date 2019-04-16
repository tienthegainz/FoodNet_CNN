from scipy.misc import imresize, imread
import numpy as np
from keras.utils import Sequence
from os import listdir
import random
from keras.utils import np_utils

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataGenerator(Sequence):

    def __init__(self, n_classes = 11,
                shuffle=True, batch_size=100, data_path, aug_path = None):
        'Initialization'
        self.data_path = data_path
        self.n_classes = n_classes
        self.shuffle = shuffle
        if aug_path == None:
            self.x = listdir(data_path)
        else:
            self.x = listdir(data_path)+listdir(aug_path)
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
        """Data has shape of 200, 200, 3"""
        # print('\nFile: ',batch_x)
        data = np.array([
                    imresize(imread(self.data_path+file_name), (200, 200))
                    for file_name in batch_x])/255.00
        labels = np_utils.to_categorical(np.array(batch_y), num_classes=self.n_classes)
        return data, labels
