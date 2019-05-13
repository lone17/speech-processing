from preprocess import *
from predict import *
from record import *
from helpers import *

model_dir = 'model'
mean, std = np.load(model_dir + '/scale.npy')
models = []
for i in range(config.num_folds):
    model = get_model()
    model_path = model_dir + '/checkpoints/best_%d.h5' % i
    # print(model_path)
    model.load_weights(model_path)
    models.append(model)

def predict_mic():
    audio = record()
    audio = process_audio(audio)
    audio = audio[None, :, :, None]
    audio = (audio - mean) / std
    for i, model in enumerate(models):
        tmp = model.predict(audio, batch_size=64, verbose=1)
        if i == 0:
            pred = tmp * 10000
        else:
            # pred = pred * tmp / (pred * tmp + (1-pred) * (1-tmp))
            pred *= tmp

    # for label, confidence in zip(idx_label, pred[0]):
    #     print(label, confidence)
    pred = np.argmax(pred, axis=1)[0]
    print(idx_label[pred])

predict_mic()
