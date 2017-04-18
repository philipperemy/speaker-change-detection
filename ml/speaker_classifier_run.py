import os
import sys

sys.path.append(os.path.abspath('..'))

from helpers.speakers_to_categorical import SpeakersToCategorical
from ml.mfcc_data_generation import generate_data
from ml.speaker_classifier import *


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


def run_model():
    data = generate_data(max_count_per_class=500)  # 10)  # 500)
    kx_train, ky_train, kx_test, ky_test, categorical_speakers = data_to_keras(data)
    m = get_model()
    build_model(m)
    fit_model(m, kx_train, ky_train, kx_test, ky_test, max_epochs=50)
    print(categorical_speakers.get_speaker_from_index(inference_model(m, kx_train[0:100])))


if __name__ == '__main__':
    run_model()
