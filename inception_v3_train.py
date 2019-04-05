from custom_CNN.inception_v3 import build_inception_v3
from process_images.data_generator import DataGenerator
from process_images.process import load_class, load_image
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
import h5py
from pathlib import Path
from keras.models import load_model

if __name__ == '__main__':
    # Init generator
    train_path = 'Food-11/training/'
    train_gen = DataGenerator(data_path=train_path)
    val_path = 'Food-11/validation/'
    val_gen = DataGenerator(data_path=val_path, batch_size=60)
    classes, indexes = load_class('class_description.txt')
    """Continue to train"""
    print('Load model\n')
    model = load_model('train_data/inception_v3.hdf5')
    model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    """New model"""
    # print('Init model\n')
    # model = build_inception_v3(len(indexes))

    checkpointer = ModelCheckpoint(filepath='train_data/inception_v3.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('train_data/inception_v3.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.001)

    """Fit models by generator"""
    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        callbacks=[csv_logger, checkpointer, reduce_lr],
                        epochs=10)
