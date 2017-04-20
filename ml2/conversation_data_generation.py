import os
import sys

sys.path.append(os.path.abspath('..'))

from audio.audio_reader import AudioReader
from constants import c

audio = AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                    sample_rate=c.AUDIO.SAMPLE_RATE,
                    speakers_sub_list=None)


def generate_conv(generate_mix_fun=audio.generate_mix):
    # 14 seconds. sentence is about 2 sec. so 7 sentences per speaker
    all_speaker_transition_task = c.AUDIO.SPEAKER_FOR_TRANSITION_TASK
    # all_speaker_transition_task = audio.get_speaker_list()
    cut = len(all_speaker_transition_task) // 2
    training_speakers = all_speaker_transition_task[0:cut]
    testing_speakers = all_speaker_transition_task[cut:]

    training_targets = []
    for training_speaker in training_speakers:
        t = audio.define_random_mix(num_sentences=20, speaker_ids_to_choose_from=[training_speaker])
        training_targets.append(t)
    tr = generate_mix_fun(sum(training_targets, []))

    testing_targets = []
    for testing_speaker in testing_speakers:
        t = audio.define_random_mix(num_sentences=20, speaker_ids_to_choose_from=[testing_speaker])
        testing_targets.append(t)
    te = generate_mix_fun(sum(testing_targets, []))
    return tr, te, c.AUDIO.SAMPLE_RATE


def generate_conv_voice_only():
    return generate_conv(generate_mix_fun=audio.generate_mix_with_voice_only)
