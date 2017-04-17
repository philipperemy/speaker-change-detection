import numpy as np
from keras.utils.np_utils import to_categorical

from ml.data_generation import generate_data
from ml.speaker_classifier import *


def data_to_keras(data):
    speaker_ids = sorted(list(data.keys()))
    int_speaker_ids = list(range(len(speaker_ids)))
    map_speakers_to_int = dict([(k, v) for (k, v) in zip(speaker_ids, int_speaker_ids)])
    speaker_categories = to_categorical(int_speaker_ids, num_classes=len(speaker_ids))

    keras_x_train = []
    keras_y_train = []
    keras_x_test = []
    keras_y_test = []
    for speaker_id in speaker_ids:
        d = data[speaker_id]
        y = speaker_categories[map_speakers_to_int[d['speaker_id']]]
        for x_train_elt in data[speaker_id]['train']:
            for x_train_sub_elt in x_train_elt:
                keras_x_train.append(x_train_sub_elt)
                keras_y_train.append(y)

        for x_test_elt in data[speaker_id]['test']:
            for x_test_sub_elt in x_test_elt:
                keras_x_test.append(x_test_sub_elt)
                keras_y_test.append(y)

    keras_x_train = np.array(keras_x_train)
    keras_x_test = np.array(keras_x_test)

    keras_y_train = np.array(keras_y_train)
    keras_y_test = np.array(keras_y_test)

    return keras_x_train, keras_y_train, keras_x_test, keras_y_test


def run_model():
    data = generate_data(max_count_per_class=500)
    kx_train, ky_train, kx_test, ky_test = data_to_keras(data)
    m = get_model()
    build_model(m)
    fit_model(m, kx_train, ky_train, kx_test, ky_test)


if __name__ == '__main__':
    run_model()
