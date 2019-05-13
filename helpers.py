import numpy as np

from keras.models import  Model
from keras.layers import (BatchNormalization, Flatten, MaxPool2D, Activation,
                          Input, Dense, Dropout, Convolution2D)
from keras import optimizers

def transform_target(y):
    y = y.apply(lambda i: label_idx[i])
    y = to_categorical(y, num_classes=6)

    return y

def reverse_transform_target(y):
    y = np.argmax(y, axis=1)
    y = y.apply(lambda i: idx_label[i])

    return y

label_idx = {'female_north': 0, 'female_central': 1, 'female_south': 2,
             'male_north': 3, 'male_central': 4, 'male_south': 5}
idx_label = ['female_north', 'female_central', 'female_south',
             'male_north', 'male_central', 'male_south']

class Config(object):
    def __init__(self,
                 sampling_rate=22050, audio_duration=10, num_classes=6,
                 num_folds=5, learning_rate=0.0003, max_epochs=100, num_mfcc=40,
                 patience=10):

        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.num_classes = num_classes
        self.num_mfcc = num_mfcc
        self.num_folds = num_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience

        self.audio_length = self.sampling_rate * self.audio_duration
        self.dim = (self.num_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)

def get_model():

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

    model = Model(inputs=inp, outputs=out)
    opt = optimizers.Nadam(config.learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model

config = Config(sampling_rate=22050, audio_duration=10, num_folds=5,
                learning_rate=0.0003, num_mfcc=40, patience=10, max_epochs=100)
