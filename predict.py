import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from helpers import *
from cnn import *

def predict(model_dir, data, use_xgboost=True):
    mean, std = np.load(model_dir + '/scale.npy')
    data = (data - mean) / std

    if use_xgboost:
        classifier = pickle.load(open('xgboost_clf_new.pickle', 'rb'))
        bottleneck = []
    else:
        pred = 10**config.num_folds

    for i in range(config.num_folds):
        model = get_model(return_bottleneck=use_xgboost)
        model_path = model_dir + '/checkpoints/best_%d.h5' % i
        print(model_path)
        model.load_weights(model_path, by_name=True)
        tmp = model.predict(data, batch_size=64, verbose=1)

        if use_xgboost:
            bottleneck.append(tmp)
        else:
            pred *= tmp

    if use_xgboost:
        bottleneck = np.hstack(bottleneck)
        dtest = xgb.DMatrix(bottleneck)
        pred = classifier.predict(dtest, ntree_limit=classifier.best_ntree_limit)

    pred = np.argmax(pred, axis=1)

    return pred

def predict_new(model_path, X):
    preds = []
    model = load_model(model_path)

    for chunks in X:
        if len (chunks) > 1:
            chunks = chunks[:-1]
        pred = None
        for chunk in chunks:
            tmp = model.predict(chunk[None, :, :, :])
            if pred is None:
                pred = tmp
            else:
                pred = pred * tmp / (pred * tmp + (1 - pred) * (1 - tmp))
        pred = np.argmax(pred, axis=1)
        preds.append(pred[0])

    return preds

def get_submission(pred, data_path='data'):
    files = sorted(os.listdir(data_path), key=lambda f: int(f.split('.')[0]))

    gender = pred // 3
    accent = pred % 3
    submit = pd.DataFrame(dict(id=files, gender=gender, accent=accent))
    submit[['id', 'gender', 'accent']].to_csv('submission.csv', index=False)

# X = np.load('voice/X_test_10s.npy')
X = pickle.load(open('X_test_chunks.pickle', 'rb'))
y = np.load('y_test_10s.npy')
# pred = predict('model', X, True)
pred = predict_new('model.63-0.51.hdf5', X)
# get_submission(pred, 'voice/public_test')

print(accuracy_score(y, pred))
print(classification_report(y, pred))
print(confusion_matrix(y, pred))
