import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from helpers import *

def predict(model_dir, data):
    mean, std = np.load(model_dir + '/scale.npy')
    data = (data - mean) / std

    for i in range(config.num_folds):
        model = get_model()
        model_path = model_dir + '/checkpoints/best_%d.h5' % i
        print(model_path)
        model.load_weights(model_path)
        tmp = model.predict(data, batch_size=64, verbose=1)
        if i == 0:
            pred = tmp * 10000
        else:
            # pred = pred * tmp / (pred * tmp + (1-pred) * (1-tmp))
            pred *= tmp

    pred = np.argmax(pred, axis=1)

    return pred

def get_submission(pred, data_path='data'):
    files = sorted(os.listdir(data_path), key=lambda f: int(f.split('.')[0]))

    gender = pred // 3
    accent = pred % 3
    submit = pd.DataFrame(dict(id=files, gender=gender, accent=accent))
    submit[['id', 'gender', 'accent']].to_csv('submission.csv', index=False)

# X = np.load('voice/X_test_10s.npy')
# y = np.load('y_test.npy')
# pred = predict('model', X)

# print(accuracy_score(y, pred))
# print(classification_report(y, pred))
# print(confusion_matrix(y, pred))
