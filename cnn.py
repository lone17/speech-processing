from keras.models import  Model
from keras.layers import (BatchNormalization, Flatten, MaxPool2D, Activation,
                          Input, Dense, Dropout, Convolution2D)
from keras import optimizers
from helpers import config

def get_model(return_bottleneck=False, input_shape=config.dim):

    inp = Input(shape=input_shape)
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

    bottleneck = Flatten()(x)
    x = Dense(64)(bottleneck)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    out = Dense(config.num_classes, activation='softmax')(x)

    if return_bottleneck:
        model = Model(inputs=inp, outputs=bottleneck)
        return model

    model = Model(inputs=inp, outputs=out)
    opt = optimizers.Nadam(config.learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model

def get_new_model(model_path=None):
    inp = Input(shape=(40, 65, 1))
    x = Convolution2D(8, (3,3), padding="valid")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(8, (3,5), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(16, (3,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(16, (3,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(32, (3,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = MaxPool2D((2,2))(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(64, (3,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (3,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Convolution2D(64, (3,5), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.5)(x)

    x = Convolution2D(16, (1,1))(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Flatten()(x)

    out = Dense(config.num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    if model_path is not None:
        model.load_weights(model_path)

    return model
