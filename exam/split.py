from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import cross_validation
import numpy as np
import pandas as pd

FX_LIST = ['EURUSD', 'USDJPY', 'GBPUSD']
FILE_PREX = '../../../data/fx'
SCALE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test_size = 0.25


def sp_24(i):
    for fx in FX_LIST:
        fs_t_path = ['%s/NFs/%s_%i.npy' % (FILE_PREX, fx, i),
                     '%s/T/%s_%i.pkl' % (FILE_PREX, fx, i)]
        fs = np.load(fs_t_path[0])
        t = pd.read_pickle(fs_t_path[1])
        f_train, f_test, t_train, t_test = cross_validation.train_test_split(
            fs, t, test_size=test_size, random_state=42)
        np.save('%s/NFs/%s_train_%i.npy' % (FILE_PREX, fx, i), f_train)
        np.save('%s/NFs/%s_test_%i.npy' % (FILE_PREX, fx, i), f_test)
        np.save('%s/NFs/%s_plot_%i.npy' % (FILE_PREX, fx, i), fs)
        t_train.to_pickle('%s/T/%s_train_%i.pkl' % (FILE_PREX, fx, i))
        t_test.to_pickle('%s/T/%s_test_%i.pkl' % (FILE_PREX, fx, i))
        t.to_pickle('%s/T/%s_plot_%i.pkl' % (FILE_PREX, fx, i))
    return len(t)


def sp_5(i, p):
    for fx in FX_LIST:
        fs_t_path = ['%s/NFs/%s_5_%i.npy' % (FILE_PREX, fx, i),
                     '%s/T/%s_5_%i.pkl' % (FILE_PREX, fx, i)]
        fs = np.load(fs_t_path[0])[:p]
        t = pd.read_pickle(fs_t_path[1])[:p]
        f_train, f_test, t_train, t_test = cross_validation.train_test_split(
            fs, t, test_size=test_size, random_state=42)
        np.save('%s/NFs/%s_train_5_%i.npy' % (FILE_PREX, fx, i), f_train)
        np.save('%s/NFs/%s_test_5_%i.npy' % (FILE_PREX, fx, i), f_test)
        np.save('%s/NFs/%s_plot_5_%i.npy' % (FILE_PREX, fx, i), fs)
        t_train.to_pickle('%s/T/%s_train_5_%i.pkl' % (FILE_PREX, fx, i))
        t_test.to_pickle('%s/T/%s_test_5_%i.pkl' % (FILE_PREX, fx, i))
        t.to_pickle('%s/T/%s_plot_5_%i.pkl' % (FILE_PREX, fx, i))


if __name__ == '__main__':
    for i in SCALE:
        sp_5(i, sp_24(i))
