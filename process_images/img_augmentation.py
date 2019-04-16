from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from os import listdir

def shift_left(img_path):
    path = img_path.split('training')
    new_name = path[1].split('.')
    new_path = path[0] + 'augmentation' + new_name[0] + '(1).' + new_name[1]
    img = Image.open(img_path)
    img = np.array(img)
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]
    CHANNEL = img.shape[2]
    # Shifting Left
    for i in range(WIDTH-1, 1, -1):
      for j in range(HEIGHT):
        if (i >= 30):
            img[j][i] = img[j][i-30]
        else:
            img[j][i] = [0, 0, 0]
    Image.fromarray(img).save(new_path)

def shift_right(img_path):
    path = img_path.split('training')
    new_name = path[1].split('.')
    new_path = path[0] + 'augmentation' + new_name[0] + '(2).' + new_name[1]
    img = Image.open(img_path)
    img = np.array(img)
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]
    CHANNEL = img.shape[2]
    # Shifting Right
    for i in range(WIDTH):
      for j in range(HEIGHT):
        if (i < WIDTH-30):
            img[j][i] = img[j][i+30]
        else:
            img[j][i] = [0, 0, 0]
    Image.fromarray(img).save(new_path)

def shift_down(img_path):
    path = img_path.split('training')
    new_name = path[1].split('.')
    new_path = path[0] + 'augmentation' + new_name[0] + '(3).' + new_name[1]
    img = Image.open(img_path)
    img = np.array(img)
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]
    CHANNEL = img.shape[2]
    # Shifting Right
    for i in range(WIDTH):
      for j in range(HEIGHT-1, 0, -1):
        if (j >= 30):
            img[j][i] = img[j-30][i]
        else:
            img[j][i] = [0, 0, 0]
    Image.fromarray(img).save(new_path)

def rotate(img_path):
    path = img_path.split('training')
    new_name = path[1].split('.')
    new_path = path[0] + 'augmentation' + new_name[0] + '(4).' + new_name[1]
    img = Image.open(img_path)
    img = img.rotate(random.randint(-45, 45))
    img.save(new_path)

def add_noise(img_path):
    path = img_path.split('training')
    new_name = path[1].split('.')
    new_path = path[0] + 'augmentation' + new_name[0] + '(5).' + new_name[1]
    img = Image.open(img_path)
    img = np.array(img)
    noise = np.random.randint(-50, 50, size = img.shape, dtype = 'int')
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]
    CHANNEL = img.shape[2]
    # add noise
    for i in range(0, HEIGHT, 3):
        for j in range(0, WIDTH, 3):
            for k in range(0, CHANNEL, 3):
                if (img[i][j][k] <= 205 and img[i][j][k] >=50):
                    img[i][j][k] += noise[i][j][k]
    Image.fromarray(img).save(new_path)

def augmentation_class(root, class_target):
    print('Augmentation the {} dataset\n'.format(root))
    imgs = listdir(root)

    for img in imgs:
        class_im = img.split('_')[0]
        if(int(class_im)==class_target):
            img = root + img
            shift_left(img)
            shift_right(img)
            shift_down(img)
            rotate(img)
            add_noise(img)

if __name__ == '__main__':
    train_path = '../Food-11/training/'
    augmentation_class(root=train_path, class_target=2)
