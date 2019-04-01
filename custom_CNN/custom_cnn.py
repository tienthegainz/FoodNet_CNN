from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

"""
This is my custom CNN model which contains 2 Conv2D, 1 AvgPool, 1 MaxPool, 1 Flatten and 1 FCL sigmoid
"""

def build_custom_CNN(n_classes):
    model = Sequential()
    model.add(
        Conv2D(filters=32,
            kernel_size=(5, 5),
            input_shape=(32, 32, 3),
            activation='relu',
            padding='same'
        )
    )
    model.add(
        AveragePooling2D(pool_size=(2, 2))
    )
    model.add(
        Conv2D(filters=32,
            kernel_size=(5, 5),
            input_shape=(16, 16, 3),
            activation='relu',
            padding='same'
        )
    )
    model.add(
        MaxPooling2D(pool_size=(2, 2))
    )
    model.add(
        Flatten()
    )
    model.add(
        Dense(
            units=n_classes,
            activation = 'sigmoid'
        )
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=False),
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    model = build_custom_CNN(11)
    print(model.summary())
