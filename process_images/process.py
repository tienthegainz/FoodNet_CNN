import os
from os import listdir
import h5py
from scipy.misc import imresize, imread
import numpy as np
from keras.utils import np_utils

def pprint(A):
    if A.ndim==1:
        print(A)
    else:
        w = max([len(str(s)) for s in A])
        print(u'\u250c'+u'\u2500'*w+u'\u2510')
        for AA in A:
            print(' ', end='')
            print('[', end='')
            for i,AAA in enumerate(AA[:-1]):
                w1=max([len(str(s)) for s in A[:,i]])
                print(str(AAA)+' '*(w1-len(str(AAA))+1),end='')
            w1=max([len(str(s)) for s in A[:,-1]])
            print(str(AA[-1])+' '*(w1-len(str(AA[-1]))),end='')
            print(']')
        print(u'\u2514'+u'\u2500'*w+u'\u2518')

def load_class(path):
    with open(path, 'r') as foods:
        classes = [food.split()[1] for food in foods]
        print('Name: {}\n'.format(classes))
        index = np.arange(len(classes))
        print('Index: {}\n'.format(index))
    return classes, index

def load_image(root, min_side=200):
    print('Loading the {} dataset\n'.format(root))
    images = []
    classes = []

    imgs = sorted(os.listdir(root)) # get items in 'root path'

    for img in listdir(root):
        # resize into 200, 200
        im = imresize(imread(root + img), (min_side, min_side))
        arr = np.array(im)/255.00
        images.append(arr)
        classes.append(img.split('_')[0])
    print('Finish loading the {} dataset\n'.format(root))

    return np.array(images, dtype=np.float), np_utils.to_categorical(np.array(classes), 11)

def load_image_per_class(root, class_target, min_side=200):
    print('Loading the {} dataset\n'.format(root))
    images = []
    classes = []

    imgs = os.listdir(root) # get items in 'root path'

    for img in listdir(root):
        # resize into 200, 200
        class_im = img.split('_')[0]
        if(int(class_im)==class_target):
            print(img, ' ', class_im, '\n')
            im = imresize(imread(root + img), (min_side, min_side))
            arr = np.array(im)/255.00
            images.append(arr)
            classes.append(class_im)
    print('Finish loading the {} dataset\n'.format(root))

    return np.array(images, dtype=np.float), np_utils.to_categorical(np.array(classes), 11)

if __name__ == "__main__":
    # Classes init
    # classes, indexes = load_class('../class_description.txt')
    test_path = '../Food-11/evaluation/'
    imgs, img_classes = load_image_per_class(test_path, 1)
    # print(imgs)
    # pprint(imgs[0])
