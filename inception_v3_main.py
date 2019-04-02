from custom_CNN.inception_v3 import build_inception_v3
from process_images.data_generator import DataGenerator
from process_images.process import load_class, load_image
from keras.callbacks import ModelCheckpoint
import h5py
from pathlib import Path
from keras.models import load_model

if __name__ == '__main__':
    train_path = 'Food-11/training/'
    train_gen = DataGenerator(data_path=train_path)
    val_path = 'Food-11/validation/'
    val_gen = DataGenerator(data_path=val_path, batch_size=100)
    print('Init model\n')
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
    from keras.optimizers import SGD
    from keras.layers import Flatten
    # Test model
    model = Sequential()
    model.add(
        Conv2D(filters=32,
            kernel_size=(5, 5),
            input_shape=(200, 200, 3),
            activation='relu',
            padding='same'
        )
    )
    model.add(
        AveragePooling2D(pool_size=(4, 4))
    )
    model.add(
        Flatten()
    )
    model.add(
        Dense(
            units=11,
            activation = 'sigmoid'
        )
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False),
        metrics=['accuracy']
    )
    print(model.summary())
    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        use_multiprocessing=True,
                        workers=6)
