print('>>>>> Loading dependencies...')
import os

import pickle
import librosa
import xgboost as xgb
import sounddevice as sd
from keras.models import load_model

import cnn
from record import *
from helpers import *
from preprocess import *
from generate_data import get_chunks
print('<<<<< Dependencies are loaded.')

print('>>>>> Loading models...')
use_xgboost = True

confidence_threshold = 0.5
max_duration = 20

cnn_model_dir = 'model'
xgboost_model = 'xgboost_clf_new.pickle'

mean, std = np.load(cnn_model_dir + '/scale.npy')

extractors = []
for i in range (config.num_folds):
    model = cnn.get_model(return_bottleneck=use_xgboost)
    weights_path = cnn_model_dir + '/checkpoints/best_%d.h5' % i
    model.load_weights(weights_path, by_name=True)
    extractors.append(model)

if use_xgboost:
    classifier = pickle.load(open(xgboost_model, 'rb'))
print('<<<<< Models are loaded.')

def predict(audio):
    audio = process_audio(audio)
    audio = audio[None, :, :, None]
    audio = (audio - mean) / std

    if use_xgboost:
        bottleneck = []
    else:
        probs = 10**len(extractors)

    for i, model in enumerate(extractors):
        tmp = model.predict(audio, verbose=0)
        if use_xgboost:
            bottleneck.append(tmp)
        else:
            probs *= tmp

    if use_xgboost:
        bottleneck = np.hstack(bottleneck)
        dtest = xgb.DMatrix(bottleneck)
        probs = classifier.predict(dtest, ntree_limit=classifier.best_ntree_limit)
    else:
        probs /= np.sum(probs)
        # print(probs)

    return probs[0]

# new_model = load_model('model.63-0.51.hdf5')
# def predict_new(audio):
#     chunks = get_chunks(audio)
#
#     if len (chunks) > 1:
#         chunks = chunks[:-1]
#     probs = None
#     for chunk in chunks:
#         chunk = librosa.feature.mfcc(chunk, sr=22050, n_mfcc=40)
#         tmp = new_model.predict(chunk[None, :, :, None])
#         if probs is None:
#             probs = tmp
#         else:
#             probs = probs * tmp / (probs * tmp + (1 - probs) * (1 - tmp))
#
#     probs /= np.sum(probs)
#
#     return probs[0]

os.system('cls')

while True:
    print('Instructions:')
    print()
    print('\tPress m to input from microphone. The program will automatically stop')
    print('\t        recording after %ds or when silence is detected.' %  max_duration)
    print('\tPress f to input from audio file. The program will then ask for the')
    print('\t        location of the audio file.')
    print('\tPress q to quit.')
    print()
    user_input = input('Please type in your selection: ')

    if user_input == 'q':
        break

    if user_input == 'm':
        audio = record(max_duration=20)
        # sd.play(audio, 22050, blocking=True)
    elif user_input == 'f':
        file_path = input('File location: ')
        if os.path.exists(file_path):
            audio, sr = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_best")
        else:
            os.system('cls')
            print('** Error ** Cannot find', file_path)
            continue
    else:
        os.system('cls')
        print('** Error ** Invalid option.')
        continue

    probs = predict(audio)
    max_idx = np.argmax(probs)

    prediction = idx_label[max_idx] if probs[max_idx] > confidence_threshold else 'unknown'

    prediction_str = 'Prediction: ' + prediction
    print(prediction_str)
    plot_prediction(probs, prediction_str)
    os.system('cls')
