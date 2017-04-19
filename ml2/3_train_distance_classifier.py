import os
import pickle
import sys
from glob import glob

import numpy as np
from keras.models import load_model
from natsort import natsorted

sys.path.append(os.path.abspath('..'))

from ml2.conversation_data_generation import generate_conv_voice_only
from ml2.classifier_data_generation import get_mfcc_features_390, normalize
from ml2.classifier_model_definition import predict, inference_model


# d of dim MxK (M is the number of frames, K the number of speakers in the first training set).
def dist(d1, d2):
    return np.sqrt(np.sum(np.square(np.subtract(np.mean(d1, axis=0), np.mean(d2, axis=0)))))


def get_true_speaker(slice_offset_max, mix):
    offsets = np.array([target['offset'] for target in mix])
    target_id = len(offsets) - np.sum(np.array(slice_offset_max <= offsets, dtype=int)) - 1
    return mix[target_id]['speaker_id']


def is_transition(slice_offset_min, slice_offset_max, mix):
    if slice_offset_min == 0:  # first voice is not a transition.
        return False
    last_speaker = None
    for mix_element in mix:
        if slice_offset_min <= mix_element['offset'] <= slice_offset_max:  # there is a change in the slice.
            return mix_element['speaker_id'] != last_speaker  # is this a transition?
        last_speaker = mix_element['speaker_id']
    return False


def process_conv(conv, t, sr, model, norm_data, categorical_speakers):
    mix, audio = conv  # no-overlap when slicing with t!
    likelihoods, is_transition_list = [], []
    indices = list(range(0, len(audio), int(t * sr)))
    for i, j in zip(indices, indices[1:]):
        audio_slice = audio[i:j]
        actual_speaker = get_true_speaker(j, mix)
        feat = get_mfcc_features_390(audio_slice, sr, max_frames=None)
        feat = normalize(feat, norm_data[actual_speaker]['mean_train'], norm_data[actual_speaker]['std_train'])
        audio_slice_is_transition = is_transition(i, j, mix)
        is_transition_list.append(audio_slice_is_transition)
        # works better with log=False. Because values are ver close to 0.0, so log diverges.
        likelihoods.append(predict(model, feat, log=False))
        predicted_speaker_id = inference_model(model, feat)
        predicted_speaker = categorical_speakers.get_speaker_from_index(predicted_speaker_id)
        print('speaker predicted = {}, actual speaker = {}'.format(predicted_speaker, actual_speaker))

    print(np.where(np.array(is_transition_list, dtype=int))[0])
    distances = [0.0]
    for i, (d1, d2) in enumerate(zip(likelihoods, likelihoods[1:])):
        d = dist(d1, d2)
        print('i = {}, transition = {}, dist = {}'.format(i, is_transition_list[i], d))
        distances.append(d)
    distances = np.array(distances)
    import matplotlib.pyplot as plt
    plt.plot(distances)
    plt.show()
    # # model_output has shape (num_slices, M, K)
    # while next_index < len(audio):


def find_optimal_threshold():
    # is the model correctly saved?
    categorical_speakers = pickle.load(open('/tmp/speaker-change-detection-categorical_speakers.pkl', 'rb'))
    norm_data = pickle.load(open('/tmp/speaker-change-detection-norm.pkl', 'rb'))
    checkpoints = natsorted(glob('checkpoints/*.h5'))
    assert len(checkpoints) != 0, 'No checkpoints found.'
    m = load_model(checkpoints[-1])
    train, test, sr = generate_conv_voice_only()
    t = 2  # seconds
    process_conv(train, t, sr, m, norm_data, categorical_speakers)
    process_conv(test, t, sr, m, norm_data, categorical_speakers)


if __name__ == '__main__':
    find_optimal_threshold()
