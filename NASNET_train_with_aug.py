from custom_CNN.NASNET import build_nasnetlarge
# from process_images.data_generator import DataGenerator
from process_images.process import load_image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
import h5py
from pathlib import Path
from keras.models import load_model
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Init generator
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
    STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
    STEP_SIZE_VALID=val_gen.n//val_gen.batch_size
    """Continue to train"""
    # print('Load model\n')
    model = load_model('train_data/mobilenet_v2_SGD.02-4.29.hdf5')
    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    """New model"""
    # print('Init model\n')
    # model = build_nasnetlarge(11)

    checkpointer = ModelCheckpoint(filepath='train_data/nasnet_sgd.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('train_data/inception_v3.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.001)

    """Fit models by generator"""
    history = model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=[csv_logger, checkpointer, reduce_lr],
                        epochs=20)
    """Plot training history"""
    # Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('history/NASNetLarge_Accuracy.png')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('history/NASNetLarge_Loss.png')
