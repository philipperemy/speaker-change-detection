import os
import sys

sys.path.append(os.path.abspath('..'))

from audio.audio_reader import AudioReader
from constants import c

speakers_sub_list = None
audio = AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                    sample_rate=c.AUDIO.SAMPLE_RATE,
                    speakers_sub_list=speakers_sub_list)


def generate_conv():
    # 14 seconds. sentence is about 2 sec. so 7 sentences per speaker
    all_speakers = audio.get_speaker_list()
    cut = len(all_speakers) // 2
    training_speakers = all_speakers[0:cut]
    testing_speakers = all_speakers[cut:]

    training_targets = []
    for training_speaker in training_speakers:
        t = audio.define_random_mix(num_sentences=6, speaker_ids_to_choose_from=[training_speaker])
        training_targets.append(t)
    tr = audio.generate_mix(sum(training_targets, []))

    testing_targets = []
    for testing_speaker in testing_speakers:
        t = audio.define_random_mix(num_sentences=6, speaker_ids_to_choose_from=[testing_speaker])
        testing_targets.append(t)
    te = audio.generate_mix(sum(testing_targets, []))
    return tr, te, c.AUDIO.SAMPLE_RATE

if __name__ == '__main__':
    generate_conv()
