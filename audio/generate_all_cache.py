import os
import sys

sys.path.append(os.path.abspath('..'))

from audio.audio_reader import AudioReader
from constants import c


def main():
    AudioReader(audio_dir=c.AUDIO.VCTK_CORPUS_PATH,
                sample_rate=c.AUDIO.SAMPLE_RATE,
                speakers_sub_list=['p225', 'p226', 'p227', 'p228'])


if __name__ == '__main__':
    main()
