import numpy as np
import os
import gc
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras import backend as K

from helpers import *
from cnn import *

def train(model_id, psuedo_labeling=False, k=-1, evaluate=False):
    model_id = str(object=model_id)

    X = np.load('X_train.npy', mmap_mode=None)
    y = np.load('y_train.npy')

    if psuedo_labeling:
        X = np.concatenate([X, np.load('X_test.npy', mmap_mode=None)], axis=0)
        y = np.concatenate([y, np.load('y_psuedo.npy')], axis=0)

    if not os.path.exists(model_id):
        os.mkdir(model_id)
        os.mkdir(model_id + '/checkpoints')

    if evaluate:
        X, X_test, y, y_test = train_test_split(X, y, test_size=6000, stratify=y)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X = (X - mean)/std
    if evaluate:
        X_test = (X_test - mean)/std

    np.save(model_id + '/scale', [mean, std])

    if evaluate:
        validate = []
        cum_pred = np.ones((len(y_test), config.num_classes)) * 100000

    y_categorical = to_categorical(y, num_classes=config.num_classes)
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True)

    if k == -1:
        folds = list(skf.split(X, y))
        joblib.dump(folds, model_id + '/folds')
    else:
        folds = joblib.load(model_id + '/folds')

    for i, (train_split, val_split) in enumerate(folds):
        if i < k:
            continue

        K.clear_session()

        X_train, y_train, X_val, y_val = X[train_split], y_categorical[train_split], X[val_split], y_categorical[val_split]

        checkpoint = ModelCheckpoint(model_id + '/checkpoints/best_%d.h5'%i,
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=config.patience)
        callbacks_list = [checkpoint, early]

        print("#"*50)
        print("Fold: ", i)

        model = get_model()
        model_path = model_id + '/checkpoints/best_%d.h5' % i
        if os.path.exists(model_path) and k != -1:
            print(model_path)
            model.load_weights(model_path)

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=callbacks_list,
                            batch_size=64, epochs=config.max_epochs)

        del X_train
        del X_val
        del y_train
        del y_val
        gc.collect()

        if evaluate:
            pred = model.predict(X_test, batch_size=64, verbose=1)
            cum_pred *= pred
            pred = np.argmax(pred, axis=1)
            score = accuracy_score(y_test, pred)
            cm = confusion_matrix(y_test, pred)
            print(score)
            print(cm)
            validate.append('Fold %d : %f' % (i, score))
            validate.append(cm)

    if evaluate:
        cum_pred = np.argmax(cum_pred, axis=1)
        score = accuracy_score(y_test, cum_pred)
        cm = confusion_matrix(y_test, cum_pred)
        print(score)
        print(cm)
        validate.append('Folded : %f' % score)
        validate.append(cm)

        joblib.dump(validate, model_id + '/validate')


weights_path = 'model.28-0.52.hdf5'
model = get_new_model()

X_train = np.load('X_train_1.5s_new.npy')[:, :, :, None]
X_val = np.load('X_val_1.5s_new.npy')[:, :, :, None]

y_train = np.load('y_train_1.5s_new.npy')
y_val = np.load('y_val_1.5s_new.npy')

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train),
                                                  y_train)
weights = [class_weights[i] for i in y_train]

y_train = to_categorical(y_train, num_classes=6)
y_val = to_categorical(y_val, num_classes=6)


checkpoint = ModelCheckpoint('checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5',
                             monitor='val_loss', verbose=1,
                             save_best_only=True)
early = EarlyStopping(monitor="loss", mode="min", patience=config.patience)
callbacks_list = [checkpoint, early]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    callbacks=callbacks_list, batch_size=64,
                    epochs=config.max_epochs, class_weight=class_weights)
