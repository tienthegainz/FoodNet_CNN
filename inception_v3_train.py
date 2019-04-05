from custom_CNN.inception_v3 import build_inception_v3
from process_images.data_generator import DataGenerator
from process_images.process import load_class, load_image
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, CSVLogger
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
    # Check for trained models
    check_point = Path("inception_v3.hdf5")
    if check_point.is_file():
        model = load_model('inception_v3.hdf5')
    else:
    # Import model
        model = build_inception_v3(len(indexes))

    checkpointer = ModelCheckpoint(filepath='inception_v3.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('inception_v3.log')

    """Fit models by generator"""
    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        callbacks=[csv_logger, checkpointer],
                        epochs=10)
