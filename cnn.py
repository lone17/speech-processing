from keras.models import  Model
from keras.layers import (BatchNormalization, Flatten, MaxPool2D, Activation,
                          Input, Dense, Dropout, Convolution2D)
from keras import optimizers
from helpers import config

def get_model(return_bottleneck=False):

    inp = Input(shape=config.dim)
    x = Convolution2D(8, (2,2), padding="valid")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(8, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(16, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(16, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((3,3))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(64, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(10, (1,1))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    out = Dense(config.num_classes, activation='softmax')(x)

    if return_bottleneck:
        model = Model(inputs=inp, outputs=x)
        return model

    model = Model(inputs=inp, outputs=out)
    opt = optimizers.Nadam(config.learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model
