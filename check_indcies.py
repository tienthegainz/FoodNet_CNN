from keras.preprocessing.image import ImageDataGenerator
import numpy as np

TRAIN_PATH = 'Food-11-subfolder/Training'
VAL_PATH = 'Food-11-subfolder/Validation'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 20,
    width_shift_range = 10,
    height_shift_range = 10,
    zoom_range=0.2,
    horizontal_flip=True)

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(200, 200),
    batch_size=100,
    class_mode='categorical')
val_gen = train_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(200, 200),
    batch_size=100,
    class_mode='categorical')
