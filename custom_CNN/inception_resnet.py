from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import keras.backend as K
from keras.utils import plot_model
# import math


# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_inception_rasnet(n_classes):
    # Clear memory for new model
    K.clear_session()
    # Put the Inception V3 (cut out the classifer part) and our custom classifier on top
    # base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(200, 200, 3)))
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(200, 200, 3)))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.8)(x)
    predictions = Dense(n_classes, init='glorot_uniform', W_regularizer=l2(.002), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)
    opt = SGD(lr=0.01, momentum=0.1, decay=1e-6, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model

if __name__ == '__main__':
    model = build_inception_rasnet(11)
    plot_model(model, to_file='../NASNetLarge.png')
