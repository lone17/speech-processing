import numpy as np
import pandas as pd
import os
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

def get_submission(model_dir, data_path='data'):
    model_dir = str(model_dir)

    X_test = np.load('X_test.npy')

    files = sorted(os.listdir(data_path), key=lambda f: int(f.split('.')[0]))

    pred = predict(model_dir, data)

    gender = pred // 3
    accent = pred % 3
    submit = pd.DataFrame(dict(id=files, gender=gender, accent=accent))
    submit[['id', 'gender', 'accent']].to_csv('submission.csv', index=False)

# X = np.load('X_train.npy')
# pred = predict('model', X)
