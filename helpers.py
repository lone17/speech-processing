import cv2
import numpy as np
import matplotlib.pyplot as plt

def transform_target(y):
    y = y.apply(lambda i: label_idx[i])
    y = to_categorical(y, num_classes=6)

    return y

def reverse_transform_target(y):
    y = np.argmax(y, axis=1)
    y = y.apply(lambda i: idx_label[i])

    return y

label_idx = {'female_north': 0, 'female_central': 1, 'female_south': 2,
             'male_north': 3, 'male_central': 4, 'male_south': 5}
idx_label = ['female_north', 'female_central', 'female_south',
             'male_north', 'male_central', 'male_south']

class Config(object):
    def __init__(self,
                 sampling_rate=22050, audio_duration=10, num_classes=6,
                 num_folds=5, learning_rate=0.0003, max_epochs=100, num_mfcc=40,
                 patience=10):

        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.num_classes = num_classes
        self.num_mfcc = num_mfcc
        self.num_folds = num_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience

        self.audio_length = self.sampling_rate * self.audio_duration
        self.dim = (self.num_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)

config = Config(sampling_rate=22050, audio_duration=10, num_folds=5,
                learning_rate=0.0003, num_mfcc=40, patience=10, max_epochs=100)

def plot_prediction(probs, prediction_str=''):
    reigion_labels = ['North', 'Central', 'South'] * 2
    gender_probs = [sum(probs[:3]), sum(probs[3:])]

    center = plt.Circle((0, 0), 0.8, fc='white')

    gender_colours = ['#ffb3e6', '#66b3ff']
    reigion_colours = plt.get_cmap('Pastel2')((np.linspace(0, 1.0, 3)))

    fig = plt.figure(figsize=(9,9))
    plt.pie(probs, colors=reigion_colours, labels=reigion_labels, autopct='%1.2f%%',
            startangle=90, pctdistance=0.85, counterclock=False, radius=2, rotatelabels=True)
    plt.pie(gender_probs, colors=gender_colours, autopct='%1.2f%%', labels=['Female', 'Male'],
            startangle=90, pctdistance=0.78, counterclock=False, radius=1.4,
            labeldistance=0.25)

    plt.gcf().gca().add_artist(center)
    plt.suptitle(prediction_str + '\n\n' + 'Press any key to continue',
                 x=0.5, y=0.1)
    plt.tight_layout()
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
