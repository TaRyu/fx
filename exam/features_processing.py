"""Predict based on cnn"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import preprocessing

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD']
FILE_PREX = '../../../data/fx'
SCALE = 10


def features_processing(file_in, file_out):
    features = np.load(file_in)
    new_features = preprocessing.minmax_i(features, axis=1)
    np.save(file_out, new_features.astype('float32'))


def pr_24(i):
    for fx in FX_LIST:
        file_in = '%s/Fs/%s_%i.npy' % (FILE_PREX, fx, i)
        file_out = '%s/NFs/%s_%i.npy' % (FILE_PREX, fx, i)
        features_processing(file_in, file_out)


def pr_5(i):
    for fx in FX_LIST:
        file_in = '%s/Fs/%s_5_%i.npy' % (FILE_PREX, fx, i)
        file_out = '%s/NFs/%s_5_%i.npy' % (FILE_PREX, fx, i)
        features_processing(file_in, file_out)


if __name__ == '__main__':
    for i in SCALE:
        pr_24(i)
        pr_5(i)
