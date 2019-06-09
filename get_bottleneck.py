import os
import numpy as np

from helpers import get_model, config


X = np.load('voice/X_test_10s.npy')
bottleneck = []

model_dir = 'model'
mean, std = np.load(model_dir + '/scale.npy')
X = (X - mean) / std

for i in range(config.num_folds):
    model = get_model(return_bottleneck=True)
    model_path = model_dir + '/checkpoints/best_%d.h5' % i
    print(model_path)
    model.load_weights(model_path, by_name=True)
    tmp = model.predict(X, batch_size=64, verbose=1)
    bottleneck.append(tmp)

bottleneck = np.hstack(bottleneck)
np.save('bottleneck_test', bottleneck)
