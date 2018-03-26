import os
import pickle

import numpy as np

from constants import c
from helpers.speakers_to_categorical import SpeakersToCategorical
from ml.classifier_model_definition import get_model, fit_model, build_model


def data_to_keras(data):
    categorical_speakers = SpeakersToCategorical(data)
    kx_train, ky_train, kx_test, ky_test = [], [], [], []
    ky_test = []
    for speaker_id in categorical_speakers.get_speaker_ids():
        d = data[speaker_id]
        y = categorical_speakers.get_one_hot_vector(d['speaker_id'])
        for x_train_elt in data[speaker_id]['train']:
            for x_train_sub_elt in x_train_elt:
                kx_train.append(x_train_sub_elt)
                ky_train.append(y)

        for x_test_elt in data[speaker_id]['test']:
            for x_test_sub_elt in x_test_elt:
                kx_test.append(x_test_sub_elt)
                ky_test.append(y)

    kx_train = np.array(kx_train)
    kx_test = np.array(kx_test)

    ky_train = np.array(ky_train)
    ky_test = np.array(ky_test)

    return kx_train, ky_train, kx_test, ky_test, categorical_speakers


def start_training():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    num_speakers = c.AUDIO.NUM_SPEAKERS_CLASSIFICATION_TASK
    data_filename = '/tmp/speaker-change-detection-data.pkl'
    assert os.path.exists(data_filename), 'Data does not exist.'
    print('Loading the inputs in memory. It might take a while...')
    data = pickle.load(open(data_filename, 'rb'))
    kx_train, ky_train, kx_test, ky_test, categorical_speakers = data_to_keras(data)
    print('Dumping info about categorical speakers for the next phase (train distance classifier..')
    pickle.dump(categorical_speakers, open('/tmp/speaker-change-detection-categorical_speakers.pkl', 'wb'))
    print('Defining model...')
    m = get_model(num_classes=num_speakers)
    print('Building model...')
    build_model(m)
    print('Fitting model...')
    fit_model(m, kx_train, ky_train, kx_test, ky_test)


if __name__ == '__main__':
    start_training()
