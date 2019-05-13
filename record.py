import numpy as np
import pyaudio
import librosa
import sounddevice as sd
# from preprocess import *
# from predict import *

RATE = 22050
CHUNK_SIZE = 1024
SILENT_THRESH = 30
FORMAT = pyaudio.paInt16

def is_silent(chunk):
    db = librosa.core.power_to_db(chunk)[0]
    # print(db)
    return db < - SILENT_THRESH

def normalize(audio):
    "Average the volume out"
    MAXIMUM = 16384
    ratio = float(MAXIMUM)/max(abs(i) for i in audio)

    r = array('h')
    for i in audio:
        r.append(int(i*ratio))
    return r

# p = pyaudio.PyAudio()
# stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True,
#                 frames_per_buffer=CHUNK_SIZE)
#
# num_silent = 0
# snd_started = False

# r = array('h')

# while 1:
#     # little endian, signed short
#     chuck
#     if byteorder == 'big':
#         snd_data.byteswap()
#     r.extend(snd_data)
#
#     silent = is_silent(snd_data)
#
#     if silent and snd_started:
#         num_silent += 1
#     elif not silent and not snd_started:
#         snd_started = True
#
#     if snd_started and num_silent > 30:
#         break

# sample_width = p.get_sample_size(FORMAT)
# stream.stop_stream()
# stream.close()
# p.terminate()
#
# r = normalize(r)
# r = trim(r)
# return sample_width, r

def record():
    recorded = []
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silence = 0
    recording = False
    print('recording...')
    while num_silence < 20:
        data = stream.read(CHUNK_SIZE)
        a = np.frombuffer(data, dtype=np.int16) / 2**15
        if is_silent(a):
            num_silence += 1
            # if recording:
            #     recorded.append(a)
        else:
            # if not recording:
            #     recording = True
            #     print('Sound detected, start recording')
            # recorded.append(a)
            num_silence = 0
        recorded.append(a)

    if len(recorded) > 0:
        print('Silence detected, stop recording')
        recorded = np.concatenate(recorded)
        # recorded, _ = librosa.effects.trim(recorded, top_db=SILENT_THRESH)
    else:
        print('No sound detected')

    stream.stop_stream()
    stream.close()
    p.terminate()

    return recorded

# audio, _ = librosa.core.load('D:\\speech processing\\voice\\public_test\\0.amr',
#                              sr=22050, res_type="kaiser_best")
# sd.play(audio, RATE, blocking=True)
# sd.play(recorded, RATE, blocking=True)

# audio = process_audio(recorded)
# print(predict('model', audio[None,:,:,None]))
