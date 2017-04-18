import os
import sys
from glob import glob

import numpy as np
from keras.models import load_model
from natsort import natsorted

sys.path.append(os.path.abspath('..'))

from ml2.conversation_data_generation import generate_conv
from ml2.classifier_data_generation import get_mfcc_features_390
from ml2.classifier_model_definition import predict, inference_model


# d of dim MxK (M is the number of frames, K the number of speakers in the first training set).
def dist(d1, d2):
    return np.sqrt(np.sum(np.square(np.subtract(np.mean(d1, axis=0), np.mean(d2, axis=0)))))


def is_transition(slice_offset_min, slice_offset_max, mix):
    if slice_offset_min == 0:  # first voice is not a transition.
        return False
    last_speaker = None
    for mix_element in mix:
        if slice_offset_min <= mix_element['offset'] <= slice_offset_max:  # there is a change in the slice.
            return mix_element['speaker_id'] != last_speaker  # is this a transition?
        last_speaker = mix_element['speaker_id']
    return False


def process_conv(conv, t, sr, model):
    mix, audio = conv  # no-overlap when slicing with t!
    mfcc_features, log_likelihoods, is_transition_list = [], [], []
    indices = list(range(0, len(audio), int(t * sr)))
    for i, j in zip(indices, indices[1:]):
        audio_slice = audio[i:j]
        feat = get_mfcc_features_390(audio_slice, sr, max_frames=None)
        audio_slice_is_transition = is_transition(i, j, mix)
        is_transition_list.append(audio_slice_is_transition)
        mfcc_features.append(feat)
        log_likelihoods.append(predict(model, feat, log=True))

    print(np.array([inference_model(model, v) for v in mfcc_features]))
    print(np.where(np.array(is_transition_list, dtype=int))[0])
    distances = [0.0]
    for i, (d1, d2) in enumerate(zip(log_likelihoods, log_likelihoods[1:])):
        d = dist(d1, d2)
        print('i = {}, transition = {}, dist = {}'.format(i, is_transition_list[i], d))
        distances.append(d)
        # import matplotlib.pyplot as plt
        # plt.plot(distances)
        # plt.show()
        # # model_output has shape (num_slices, M, K)
        # while next_index < len(audio):


def find_optimal_threshold():
    # all_t = [0.5, 1.0, 2.0]
    # t = all_t[0]
    checkpoints = natsorted(glob('checkpoints/*.h5'))
    assert len(checkpoints) != 0, 'No checkpoints found.'
    m = load_model(checkpoints[-1])
    train, test, sr = generate_conv()
    t = 0.5  # seconds
    process_conv(train, t, sr, m)
    process_conv(test, t, sr, m)


if __name__ == '__main__':
    find_optimal_threshold()
