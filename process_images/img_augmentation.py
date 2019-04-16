from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from os import listdir

def shift_left(img_path):
    #path = img_path.split('training')
    #new_name = path[1].split('.')
    #new_path = path[0] + 'augmentation' + new_name[0] + '(1).' + new_name[1]
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
    return img

def shift_right(img_path):
    #path = img_path.split('training')
    #new_name = path[1].split('.')
    #new_path = path[0] + 'augmentation' + new_name[0] + '(2).' + new_name[1]
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
    return img

def shift_down(img_path):
    #path = img_path.split('training')
    #new_name = path[1].split('.')
    #new_path = path[0] + 'augmentation' + new_name[0] + '(3).' + new_name[1]
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
    return img

def rotate(img_path):
    #path = img_path.split('training')
    #new_name = path[1].split('.')
    #new_path = path[0] + 'augmentation' + new_name[0] + '(4).' + new_name[1]
    img = Image.open(img_path)
    img = img.rotate(random.randint(-45, 45))
    return np.array(img)

def add_noise(img_path):
    #path = img_path.split('training')
    #new_name = path[1].split('.')
    #new_path = path[0] + 'augmentation' + new_name[0] + '(5).' + new_name[1]
    img = Image.open(img_path)
    img = np.array(img)
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]
    CHANNEL = img.shape[2]
    # add noise
    for i in range(0, HEIGHT, 3):
        for j in range(0, WIDTH, 3):
            for k in range(0, CHANNEL, 3):
                if (img[i][j][k] <= 205 and img[i][j][k] >=50):
                    img[i][j][k] += random.randint(-50, 50)
    return img

def augmentation_img(img, num = 1):
    if num == 1:
        return shift_left(img)
    elif num == 2:
        return shift_right(img)
    elif num == 3:
        return shift_down(img)
    elif num == 4:
        return rotate(img)
    elif num == 5:
        return add_noise(img)
            # Image.fromarray(img).save(new_path)

if __name__ == '__main__':
    train_path = '../Food-11/training/'
    augmentation_class(root=train_path, class_target=8)
