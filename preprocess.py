import os
import librosa
import numpy as np

from helpers import *

def read_audio(file_path):
    audio, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_best")

    return audio

def process_audio(audio):
    # Random offset / Padding
    input_length = config.audio_length
    if len(audio) > input_length:
        max_offset = len(audio) - input_length
        offset = np.random.randint(max_offset)
        audio = audio[offset:(input_length+offset)]
    else:
        if input_length > len(audio):
            max_offset = input_length - len(audio)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        audio = np.pad(audio, (offset, input_length - len(audio) - offset), "constant")

    audio = librosa.feature.mfcc(audio, sr=config.sampling_rate, n_mfcc=config.num_mfcc)

    return audio

def prepare_data(data_path='audio'):

    files = sorted(os.listdir(data_path), key=lambda f: int(f.split('.')[0]))

    X = np.empty(shape=(len(files), *config.dim))


    for i, file in enumerate(files):
        file_path = os.path.join(data_path, file)
        print(i, file_path)
        audio = read_audio(file_path)
        audio = process_audio(audio)
        audio = np.expand_dims(audio, axis=-1)

        X[i,] = audio

    return X
