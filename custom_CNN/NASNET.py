from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.nasnet import NASNetLarge
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
from keras.utils import plot_model
# import math


# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_nasnetlarge(n_classes):
    # Clear memory for new model
    K.clear_session()
    # Put the Inception V3 (cut out the classifer part) and our custom classifier on top
    # base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(200, 200, 3)))
    base_model = NASNetLarge(weights='imagenet', include_top=False, input_tensor=Input(shape=(200, 200, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    predictions = Dense(n_classes, kernel_initializer='glorot_uniform',
                        kernel_regularizer=l2(.0005), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    opt = SGD(lr=.01, momentum=.9, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = build_nasnetlarge(11)
    plot_model(model, to_file='../NASNetLarge.png')
