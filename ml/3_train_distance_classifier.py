import pickle
from glob import glob

import numpy as np
from keras.models import load_model
from natsort import natsorted
from sklearn.metrics import f1_score

from ml.classifier_data_generation import get_mfcc_features_390, normalize
from ml.classifier_model_definition import predict, inference_model
from ml.conversation_data_generation import generate_conv_voice_only


# d of dim MxK (M is the number of frames, K the number of speakers in the first training set).
def dist(d1, d2):
    return np.sqrt(np.sum(np.square(np.subtract(np.mean(d1, axis=0), np.mean(d2, axis=0)))))


def get_true_speaker(slice_offset_max, mix):
    offsets = np.array([target['offset'] for target in mix])
    target_id = len(offsets) - np.sum(np.array(slice_offset_max <= offsets, dtype=int)) - 1
    return mix[target_id]['speaker_id']


def process_conv_testing(conv, t, sr, model, norm_data, categorical_speakers, cutoff_svc):
    # fucking big code in common. TODO remove this shit
    mix, audio = conv  # no-overlap when slicing with t!
    likelihoods, is_transition_list = [], []
    indices = list(range(0, len(audio), int(t * sr)))
    prev_speaker = None
    for i, j in zip(indices, indices[1:]):
        audio_slice = audio[i:j]
        actual_speaker = get_true_speaker(j, mix)
        feat = get_mfcc_features_390(audio_slice, sr, max_frames=None)
        feat = normalize(feat, norm_data[actual_speaker]['mean_train'], norm_data[actual_speaker]['std_train'])
        if prev_speaker is None:
            audio_slice_is_transition = False
        else:
            audio_slice_is_transition = (prev_speaker != actual_speaker)
        is_transition_list.append(audio_slice_is_transition)
        # works better with log=False. Because values are ver close to 0.0, so log diverges.
        likelihoods.append(predict(model, feat, log=False))
        predicted_speaker_id = inference_model(model, feat)
        predicted_speaker = categorical_speakers.get_speaker_from_index(predicted_speaker_id)
        # print('speaker predicted = {}, actual speaker = {}'.format(predicted_speaker, actual_speaker))
        prev_speaker = actual_speaker

    is_transition_list = np.array(is_transition_list, dtype=int)
    # print(np.where(is_transition_list)[0])
    predicted_transitions = np.zeros(is_transition_list.shape)
    distances = [0.0]
    for i, (d1, d2) in enumerate(zip(likelihoods, likelihoods[1:])):
        d = dist(d1, d2)
        # print('i = {}, transition = {}, dist = {}'.format(i, is_transition_list[i], d))
        svm_pred = cutoff_svc.predict(d)[0]
        if svm_pred:  # clearer
            try:
                predicted_transitions[i + 1] = 1.0
            except IndexError:
                print('INDEX ERROR')
                pass
        distances.append(d)
    distances = np.array(distances)

    from time import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(distances)
    plt.plot(is_transition_list)
    plt_filename = '/tmp/distance_test_{}.png'.format(str(int(time())))
    plt.savefig(plt_filename)
    plt.close(fig)

    print('true = {}'.format(np.where(is_transition_list)[0]))
    print('pred = {}'.format(np.where(predicted_transitions)[0]))

    return is_transition_list, predicted_transitions


def process_conv_training(conv, t, sr, model, norm_data, categorical_speakers):
    mix, audio = conv  # no-overlap when slicing with t!
    likelihoods, is_transition_list = [], []
    indices = list(range(0, len(audio), int(t * sr)))
    prev_speaker = None
    for i, j in zip(indices, indices[1:]):
        audio_slice = audio[i:j]
        actual_speaker = get_true_speaker(j, mix)
        feat = get_mfcc_features_390(audio_slice, sr, max_frames=None)
        feat = normalize(feat, norm_data[actual_speaker]['mean_train'], norm_data[actual_speaker]['std_train'])
        if prev_speaker is None:
            audio_slice_is_transition = False
        else:
            audio_slice_is_transition = (prev_speaker != actual_speaker)
        is_transition_list.append(audio_slice_is_transition)
        # works better with log=False. Because values are ver close to 0.0, so log diverges.
        likelihoods.append(predict(model, feat, log=False))
        predicted_speaker_id = inference_model(model, feat)
        predicted_speaker = categorical_speakers.get_speaker_from_index(predicted_speaker_id)
        print('speaker predicted = {}, actual speaker = {}'.format(predicted_speaker, actual_speaker))
        prev_speaker = actual_speaker

    # print(np.where(np.array(is_transition_list, dtype=int))[0])
    distances = [0.0]
    for i, (d1, d2) in enumerate(zip(likelihoods, likelihoods[1:])):
        d = dist(d1, d2)
        # print('i = {}, transition = {}, dist = {}'.format(i, is_transition_list[i], d))
        distances.append(d)
    distances = np.array(distances)
    is_transition_list = np.array(is_transition_list)
    from time import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(distances)
    plt.plot(is_transition_list)
    plt_filename = '/tmp/distance_train_{}.png'.format(str(int(time())))
    plt.savefig(plt_filename)
    plt.close(fig)

    ground_truth_transition = is_transition_list.argmax()
    predicted_transition = distances.argmax()

    is_correct = (ground_truth_transition == predicted_transition)

    if is_correct:
        transition_distance_values = [distances[predicted_transition]]
        mask = np.ones(distances.shape, dtype=bool)
        mask[predicted_transition] = False
        no_transition_distance_values = distances[mask]
    else:
        transition_distance_values = []
        no_transition_distance_values = distances.copy()

    return is_correct, transition_distance_values, no_transition_distance_values


def find_best_cutoff(pos_values, neg_values):
    x = []
    y = []
    for pos in pos_values:
        x.append(pos)
        y.append(1)
    for neg in neg_values:
        x.append(neg)
        y.append(0)
    x = np.expand_dims(np.array(x), axis=1)
    y = np.array(y)
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(x, y)
    print('accuracy of SVC = {}'.format(clf.score(x, y)))
    # print(clf.predict(2.0)[0])
    return clf


def run_train_distance_classifier():
    # is the model correctly saved?
    categorical_speakers = pickle.load(open('/tmp/speaker-change-detection-categorical_speakers.pkl', 'rb'))
    norm_data = pickle.load(open('/tmp/speaker-change-detection-norm.pkl', 'rb'))
    checkpoints = natsorted(glob('checkpoints/*.h5'))
    assert len(checkpoints) != 0, 'No checkpoints found.'
    checkpoint_file = checkpoints[-1]
    print('Loading [{}]'.format(checkpoint_file))
    m = load_model(checkpoint_file)

    '''
    CONFIGURATION IS HERE
    '''
    t = 2  # seconds
    training_steps = 1000
    test_every_steps = 20
    # testing_steps = 100
    '''
    '''
    # POS = transition distance values
    # NEG = no transition distance values

    print('*' * 80)
    print('t = {}'.format(t))
    print('*' * 80)

    pos_list = []
    neg_list = []
    accuracy_train_list = []
    pred_list = []
    true_list = []
    for jj in range(training_steps):
        print('-' * 80)
        print('iteration = {}'.format(jj))

        train, _, sr = generate_conv_voice_only(speakers_len_training=2,
                                                speakers_len_testing=0)
        acc_train, pos_values, neg_values = process_conv_training(train, t, sr, m,
                                                                  norm_data,
                                                                  categorical_speakers)
        pos_list.extend(pos_values)
        neg_list.extend(neg_values)
        accuracy_train_list.append(acc_train)
        print('running pos mean = {}'.format(np.mean(pos_list)))
        print('running neg mean = {}'.format(np.mean(neg_list)))
        print('running accuracy train = {}'.format(np.mean(accuracy_train_list)))

        if jj != 0 and jj % test_every_steps == 0:
            cutoff_svc = find_best_cutoff(pos_list, neg_list)

            _, test, sr = generate_conv_voice_only(speakers_len_training=0,
                                                   speakers_len_testing=None)  # take them all
            true, pred = process_conv_testing(test, t, sr, m, norm_data, categorical_speakers, cutoff_svc)
            pred_list.extend(pred)
            true_list.extend(true)
            acc_test = np.mean(np.array(pred_list, dtype=int) == np.array(true_list, dtype=int))
            print('running accuracy test = {}'.format(acc_test))
            print('running f1 score = {}'.format(np.mean(f1_score(true_list, pred_list))))


if __name__ == '__main__':
    run_train_distance_classifier()
