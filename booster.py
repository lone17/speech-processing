import os

import pickle
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight

X = np.load('bottleneck_train.npy')
y = np.load('y_train.npy')
X_test = np.load('bottleneck_test.npy')
y_test = np.load('y_test.npy')

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=17,
                                                  stratify=y)

# def xgboost():

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
weights = [class_weights[i] for i in y_train]

dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

params = {
    'predictor': 'cpu_predictor',
    'objective': 'multi:softprob',
    'num_class': 6,
    'max_depth': 10
}

evallist = [(dtrain, 'train'), (dval, 'eval')]

bst = xgb.train(params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                evals=evallist,
                early_stopping_rounds=16,
                verbose_eval=True)

y_pred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
y_pred = np.argmax(y_pred, axis=1)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# xgb.plot_importance(bst)

# bst.save_model('xgboost_clf.model')
# pickle.dump(bst, open('xgboost_clf.pickle', 'wb'))
#
# del y_pred
#
# bst2 = pickle.load(open('xgboost_clf.pickle', 'rb'))
# y_pred = bst2.predict(dtest, ntree_limit=bst2.best_ntree_limit)
# y_pred = np.argmax(y_pred, axis=1)
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
