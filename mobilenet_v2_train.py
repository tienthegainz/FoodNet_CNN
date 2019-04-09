from custom_CNN.mobilenet_v2 import build_mobilenet_v2
from process_images.data_generator import DataGenerator
from process_images.process import load_class, load_image
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam, SGD
import h5py
from pathlib import Path
from keras.models import load_model
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Init generator
    train_path = 'Food-11/training/'
    train_gen = DataGenerator(data_path=train_path, n_classes=11)
    val_path = 'Food-11/validation/'
    # X_val, Y_val = load_image(val_path)
    val_gen = DataGenerator(data_path=val_path, n_classes=11)
    classes, indexes = load_class('class_description.txt')
    """Continue to train"""
    print('Load model\n')
    model = load_model('train_data/mobilenet_v2_SGD.02-4.29.hdf5')

    """New model"""
    # print('Init model\n')
    # model = build_mobilenet_v2(11)

    # opt = SGD(lr=.01, momentum=.9, decay=1e-6)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='train_data/mobilenet_v2_Adam.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('train_data/inception_v3.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.001)

    """Fit models by generator"""
    history = model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        callbacks=[csv_logger, checkpointer, reduce_lr],
                        epochs=10)
    """Plot training history"""
    # Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('history/Accuracy_MobNet_SGD.png')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('history/Loss_MobNet_SGD.png')
