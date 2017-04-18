import os
import pickle
import sys

sys.path.append(os.path.abspath('..'))

from ml2.classifier_data_generation import generate_data

if __name__ == '__main__':
    data_filename = '/tmp/speaker-change-detection-data.pkl'
    print('Data filename = {}'.format(data_filename))
    if not os.path.exists(data_filename):
        print('Data does not exist. Generating it now.')
        data = generate_data(max_count_per_class=1000)
        pickle.dump(data, open(data_filename, 'wb'))
    else:
        print('Data found. No generation is necessary.')
