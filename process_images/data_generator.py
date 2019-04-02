from scipy.misc import imresize, imread
import numpy as np
from keras.utils import Sequence
from os import listdir
import random
from keras.utils import np_utils

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class DataGenerator(Sequence):

    def __init__(self,data_path,
                class_path='class_description.txt', shuffle=True,
                batch_size=200):
        'Initialization'
        self.classes, self.indexes = self.__loadclass(path=class_path)
        self.data_path = data_path
        self.n_classes = len(self.indexes)
        self.shuffle = shuffle
        self.x = listdir(data_path)
        self.y = [img.split('_')[0] for img in self.x]
        self.batch_size=batch_size

    def __loadclass(self, path):
        with open(path, 'r') as foods:
            classes = [food.split()[1] for food in foods]
            indexes = np.arange(len(classes))
        return classes, indexes

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        data = np.array([
                    imresize(imread(self.data_path+file_name), (200, 200))
                    for file_name in batch_x])/255.00
        labels = np_utils.to_categorical(np.array(batch_y), num_classes=self.n_classes)
        return data, labels
