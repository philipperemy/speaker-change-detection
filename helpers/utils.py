import json
import os
import random

import librosa
import numpy as np
import sklearn.cluster
import sklearn.manifold
import tfboost.tensor as tensor

from constants import c
from helpers.logger import Logger

logger = Logger.instance()


def accept_with_prob(probability):
    return random.random() < probability


# test_results = core.evaluate(create_tensor_dict(['RNN/Logits/Reshape_1', 'Loss/Variable']),
#                             test_inputs)
# outputs = test_results['RNN/Logits/Reshape_1']
# metric_vector = np.ndarray.flatten(test_results['Loss/Variable'])
# print(metric_vector)
# plot_tsne(outputs[0, :, :], speaker_ids[0, :], metric_vector)

def plot_tsne(X, speaker_ids, metric_vector):
    first_speaker = speaker_ids[0]
    speaker_ids = np.array([int(speaker_id == first_speaker) for speaker_id in speaker_ids])

    metric_distance_function = create_metric_distance_function(metric_vector)

    kmeans = sklearn.cluster.KMeans(n_clusters=2, max_iter=1000)
    predicted_indices = np.array(kmeans.fit_predict(X))

    correct_prediction = predicted_indices == speaker_ids
    print(np.sum(correct_prediction.astype(np.float32)) / correct_prediction.size)

    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=40.0, metric=metric_distance_function)
    X_tsne = tsne.fit_transform(X)

    print('Plotting...')

    blue_id = speaker_ids[0]
    speaker_colors = [('b' if speaker_id == blue_id else 'r') for speaker_id in speaker_ids]

    import matplotlib.pyplot as plt
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=speaker_colors)
    plt.show()


def create_metric_distance_function(metric):
    metric_transposed = metric.transpose()

    def metric_function(v1, v2):
        return np.sqrt(np.sum(np.square(np.dot(metric_transposed, v1 - v2))))

    return metric_function


def show_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image, vmin=-100, vmax=150)
    plt.show()


def plot_many(images, titles, vmin, vmax):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=10, ncols=10)

    all_axes = axes.flatten()

    for i, image in enumerate(images[:100]):
        all_axes[i].imshow(image.transpose(), vmin=vmin, vmax=vmin)
        if titles is not None:
            all_axes[i].set_title(titles[i])

    import matplotlib.pyplot as plt
    plt.show()


def pad_transitions(arr):
    is_transition = 1  # convention!
    assert np.mean(arr) < 0.2  # just a check to make sure that this is sparse.
    # convention is 0: no transition, 1 there is transition.
    out_arr = np.array(arr)
    radius = 2
    cur = radius
    while cur < len(arr):
        val = arr[cur]
        if val == is_transition:
            for j in range(-radius, radius + 1):
                out_arr[cur + j] = is_transition
            cur += radius
            continue
        cur += 1
    return out_arr


def create_tensor_dict(tensor_names):
    return {name: tensor.by_name(name) for name in tensor_names}


def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}


def remove_keys_from_dict(keys_to_remove, dictionary):
    return {key: value for key, value in dictionary.items() if key not in keys_to_remove}


def speakers_transitions(speaker_ids):
    """`speaker_ids` has shape (batch, conversation)."""

    speaker_ids1 = speaker_ids[:-1, :]  # works for more than just batch size = 2
    speaker_ids2 = speaker_ids[1:, :]

    compares = np.array(speaker_ids1 != speaker_ids2, dtype=np.float32)
    out = compares
    # out = np.array(list(map(pad_transitions, list(compares))))
    return out


def tfboost_aligned_print(**kwargs):
    """
    Prints on a single line formatted data, each named argument must be a number (float or integer, in both standard
    or NumPy variants), a list of numbers, a dictionary of strings to numbers, or nested structures of these.
    """
    from tfboost.core import format_results
    format_list = []
    data_list = []
    for name in sorted(kwargs):
        format_list.append(name + ": {} ")
        data_list.append(format_results(kwargs[name]))
    logger.info("│ ".join(format_list).format(*data_list))


def debug_persist_inputs(x, y, _id, signal, mix):
    return debug_persist_inputs(x, y, _id, [signal], [mix])


def debug_persist_inputs_multi(x, y, _id, signals, mixes, persist_directory):
    _id = str(_id[0])
    debug_dir = os.path.join(c.GENERAL.PROJECT_DIR, persist_directory)
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    file_pattern = '{}/{}'.format(debug_dir, _id)
    for i, signal in enumerate(signals):
        librosa.output.write_wav('{}_signal_{}.wav'.format(file_pattern, i), signal, sr=c.AUDIO.SAMPLE_RATE)
    np.save('{}_x.npy'.format(file_pattern), x)
    np.save('{}_y.npy'.format(file_pattern), y)
    for i, mix in enumerate(mixes):
        with open('{}_mix_{}.json'.format(file_pattern, i), 'w') as fp:
            json.dump(mix, fp, indent=4)
