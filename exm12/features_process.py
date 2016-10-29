"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import preprocessing

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD',
           'AUDUSD', 'EURJPY', 'EURGBP']
FILE_PREX = '../../../data/fx'


def features_processing(file_in, file_out):
    features = np.load(file_in)
    new_features = preprocessing.minmax_scale(features, axis=1)
    np.save(file_out, new_features.astype('float32'))


if __name__ == '__main__':
    for fx in FX_LIST:
        file_in = '%s/Fs/%s_2H.npy' % (FILE_PREX, fx)
        file_out = '%s/NFs/%s_2H.npy' % (FILE_PREX, fx)
        features_processing(file_in, file_out)
